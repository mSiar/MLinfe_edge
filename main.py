import random
import this
import time
import numpy as np
import pandas as pd
import math
import copy
import csv
from itertools import islice
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from setting import NUM_REQUEST, UTILIZATION_THRESHOLD, MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY, MIN_RESPONSE_TIME, MAX_RESPONSE_TIME, TIME_SLOT, FLOPS, FLOPS_WATT, WATTS


class MLModel:
    def __init__(self, name, required_flops, accuracy):
        self.name = name
        self.required_flops = required_flops
        self.accuracy = accuracy #this is not dependent on allocated FLOPs, hence static


class Replica:

    def __init__(self, allocated_flops, model: MLModel, edge_node):
        self.edge_node = edge_node
        self.model = model
        self.allocated_flops = allocated_flops  
        self.busy_until = 0
        

    def process_request(self, current_time):
        exec_time = self.estimate_exec_time()
        start_time = max(current_time, self.busy_until)
        finish_time = start_time+exec_time
        self.busy_until = finish_time
        return start_time, finish_time, self.model.accuracy

    def estimate_exec_time(self):
        return self.model.required_flops/self.allocated_flops

    def energy_consumption(self):
         return self.allocated_flops/self.edge_node.flops_watt


class EdgeNode:
    def __init__(self, id, flops_capacity, flops_watt, energy_limit, models):
        self.id = id
        self.total_flops_capacity = flops_capacity
        self.flops_watt = flops_watt
        self.total_energy_limit = energy_limit
        self.available_flops = flops_capacity*UTILIZATION_THRESHOLD  
        self.available_energy = energy_limit
        self.replicas = []
        self.models = models

    def can_allocate(self, flops_needed):

        #total_allocated = sum(rep.allocated_flops for rep in self.replicas)
        #return (total_allocated + flops_needed) <= 0.8 * self.flops_capacity
        return flops_needed <= self.available_flops

    def current_energy_usage(self):
        used_energy = 0
        return sum(rep.energy_consumption() for rep in self.replicas)

    def under_energy_limit(self):
        return self.current_energy_usage() <= self.energy_limit

    def used_flops(self):
        used_flops = 0
        for replica in self.replicas:
            used_flops += replica.allocated_flops
        return used_flops 

    def used_energy(self):
        return self.total_energy_limit - self.available_energy

    def add_replica(self, model): 
        total_model_flops = sum(m.required_flops for m in self.models)
        model_weight = model.required_flops / total_model_flops
        flops = model_weight*self.total_flops_capacity*UTILIZATION_THRESHOLD 
        estimated_energy = flops/self.flops_watt
        if self.can_allocate(flops) and self.available_energy >= estimated_energy:
            replica = Replica(flops, model, self)
            self.replicas.append(replica)
            self.available_flops -= flops
            self.available_energy -= estimated_energy
            return replica

        return None

    def remove_idle_replicas(self, current_time):
        active_replicas = []
        for rep in self.replicas:
            if rep.busy_until > current_time:
                active_replicas.append(rep)
            else:
                self.available_flops += rep.allocated_flops
                self.available_energy += rep.energy_consumption()
        self.replicas = active_replicas

class Request_set:
    def __init__(self, id, arrival_time, qos_accuracy, qos_response_time, num_completed_request, estimated_accuracy, finish_time, flag):  #num_request is the number of assigned requests in the set,       estimated_accuracy is the accuracy of the ---> initialized []
        self.id = id
        self.arrival_time = arrival_time
        self.num_completed_request = num_completed_request  #Initially zero
        self.estimated_accuracy = estimated_accuracy
        self.finish_time = finish_time
        self.required_accuracy = qos_accuracy
        self.qos_response_time = qos_response_time
        self.flag = False


class DecisionMaker_energy_priority:
    def __init__(self):
        pass
    
    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs):
        replicas_list = []
        total_num_reqs = 0
        total_num_reqs = sum([NUM_REQUEST for r in request_sets])
        total_model_flops = sum(m.required_flops for m in models)
        flag = False
        cc = 0
        for node in edge_nodes:
            for model in models:
                model_weight = model.required_flops / total_model_flops
                flops = model_weight * node.total_flops_capacity * UTILIZATION_THRESHOLD #CHECK: UTILIZATION_THRESHOLD ?
                energy = flops / node.flops_watt
                if node.can_allocate(flops) and node.available_energy >= energy:
                    cc += 1
                    replicas_list.append({"edge_id": node.id,       #List of all potential containers CTN_j,l the system
                                          "model": model,
                                          "accuracy": model.accuracy,
                                          "service_time": model.required_flops/flops,
                                          "replica_flops":flops,
                                          "energy": energy,
                                          "current_replicas": len(node.replicas)
                                         })    
        
        replicas_list.sort(key=lambda r:r['energy'])
        for ctn in replicas_list:
            matching_requests = [
            r for r in request_sets
            if (NUM_REQUEST - r.num_completed_request) > 0 and
            round(sum(r.estimated_accuracy)+ctn["accuracy"], max(decimal_places(r.required_accuracy), decimal_places(ctn["accuracy"]))) >= round(r.required_accuracy*(r.num_completed_request+1), decimal_places(r.required_accuracy))] # request sets with some uncompleted requests
            if not matching_requests:
                edge_nodes[ctn["edge_id"]].remove_idle_replicas(min(req.arrival_time for req in request_sets) ) #just removing idle replicas from container
                continue
                
            edge_nodes[ctn["edge_id"]].remove_idle_replicas(min(req.arrival_time for req in matching_requests))   #remove_idle_replicas
            dominated_request = max(matching_requests,key=lambda r: (NUM_REQUEST - r.num_completed_request) / r.qos_response_time) # The set with maximum number of uncompleted requests and minimum deadline

            arrivals = [r.arrival_time for r in matching_requests]
            if len(arrivals)>=2:
                inter_arrival_times = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
            elif len(arrivals)==1:
                inter_arrival_times = arrivals
            num_replica = decide_provisioning(dominated_request, inter_arrival_times, ctn, edge_nodes)  # Provisioning decision
            
            if num_replica>0:
                matching_requests.sort(key=lambda r:r.arrival_time)
        
                new_replicas = []
                for _ in range(num_replica):
                    replica = edge_nodes[ctn["edge_id"]].add_replica(ctn["model"])
                    if replica:
                        new_replicas.append(replica)
                for r_set in matching_requests:
                    for replica in new_replicas:
                        for req in range(NUM_REQUEST-r_set.num_completed_request):
                            start, finish, accuracy = replica.process_request(r_set.arrival_time)
                            if round(finish, decimal_places(r_set.qos_response_time)) <= round(r_set.arrival_time+r_set.qos_response_time,decimal_places(r_set.qos_response_time))  and round(accuracy+sum(r_set.estimated_accuracy),max(decimal_places(accuracy), decimal_places(r_set.estimated_accuracy))) >= round(r_set.required_accuracy*(r_set.num_completed_request+1), decimal_places(r_set.required_accuracy)):
                                completed_requests.append((r_set.arrival_time, start, finish))
                                response_time = finish - r_set.arrival_time
                                response_logs.append({
                                    "request_set_id": r_set.id,
                                    "arrival_time": r_set.arrival_time,
                                    "start_time": start,
                                    "finish_time": finish,
                                    "response_time": response_time,
                                    "model": replica.model.name,
                                    "edge_node_id": replica.edge_node.id
                                })
                                r_set.estimated_accuracy.append(accuracy)
                                r_set.finish_time = max(r_set.finish_time, finish)
                                r_set.num_completed_request +=1
                

# class DecisionMaker_hungarian:
#     def __init__(self):
#         pass

#     def compute_cost(self, replica, request):
#         est_exec_time = replica.estimate_exec_time()
#         start_time = max(request.arrival_time, replica.busy_until)
#         finish_time = start_time + est_exec_time

#         projected_accuracy = sum(request.estimated_accuracy) + replica.model.accuracy
#         accuracy_budget = (request.num_completed_request + 1) * request.required_accuracy

#         # Check feasibility
#         if finish_time > request.qos_response_time + 1.0 or projected_accuracy < accuracy_budget:
#             return np.inf

#         # ---- QoS/Urgency-aware cost ----
#         slack = max(1e-3, request.qos_response_time - finish_time)
#         urgency_penalty = 1.0 / slack  # higher when tight deadline

#         # ---- Energy + node utilization penalty ----
#         energy = replica.allocated_flops / replica.edge_node.flops_watt
#         utilization_ratio = replica.edge_node.used_flops() / replica.edge_node.total_flops_capacity
#         load_penalty = 2.0 * utilization_ratio  # Weight can be tuned

#         # ---- Accuracy gain (for progress) ----
#         accuracy_gain = projected_accuracy - sum(request.estimated_accuracy)
#         value = accuracy_gain / (energy + 1e-6)

#         # ---- Final cost (lower is better) ----
#         cost = -value + urgency_penalty + load_penalty
#         return cost

#     def assign_requests_to_replicas_with_hungarian(self, request_sets, replicas):
#         num_replicas = len(replicas)
#         num_requests = len(request_sets)
#         cost_matrix = np.full((num_replicas, num_requests), fill_value=np.inf)

#         for i, replica in enumerate(replicas):
#             for j, request in enumerate(request_sets):
#                 cost_matrix[i][j] = self.compute_cost(replica, request)

#         if np.all(np.isinf(cost_matrix)):
#             return []

#         row_mask = ~np.all(np.isinf(cost_matrix), axis=1)
#         col_mask = ~np.all(np.isinf(cost_matrix), axis=0)

#         if not row_mask.any() or not col_mask.any():
#             return []

#         reduced_cost_matrix = cost_matrix[np.ix_(row_mask, col_mask)]
#         if reduced_cost_matrix.size == 0 or not np.all(np.isfinite(reduced_cost_matrix)):
#             return []

#         reduced_row_ind, reduced_col_ind = linear_sum_assignment(reduced_cost_matrix)
#         row_indices = np.where(row_mask)[0]
#         col_indices = np.where(col_mask)[0]

#         return [(row_indices[r], col_indices[c]) for r, c in zip(reduced_row_ind, reduced_col_ind)]

#     def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs):
#         replicas_list = []
#         total_model_flops = sum(m.required_flops for m in models)

#         for node in edge_nodes:
#             for model in models:
#                 model_weight = model.required_flops / total_model_flops
#                 flops = model_weight * node.total_flops_capacity * UTILIZATION_THRESHOLD
#                 energy = flops / node.flops_watt
#                 if node.can_allocate(flops) and node.available_energy >= energy:
#                     replicas_list.append({
#                         "edge_id": node.id,
#                         "model": model,
#                         "accuracy": model.accuracy,
#                         "service_time": model.required_flops / flops,
#                         "replica_flops": flops,
#                         "energy": energy,
#                     })

#         #replicas_list.sort(key=lambda r: r["energy"] / r["accuracy"])
#         replicas_list.sort(
#             key=lambda r: r["energy"] * (1 + (1 - edge_nodes[r["edge_id"]].used_flops() / (
#                         edge_nodes[r["edge_id"]].total_flops_capacity * UTILIZATION_THRESHOLD)))
#         )

#         uncompleted_requests = [r for r in request_sets if r.num_completed_request < NUM_REQUEST]

#         for ctn in replicas_list:
#             edge = edge_nodes[ctn["edge_id"]]
#             matching_requests = [
#                 r for r in uncompleted_requests
#                 if round(sum(r.estimated_accuracy),2) + ctn["accuracy"] >=
#                    (r.num_completed_request + 1) * r.required_accuracy
#             ]
#             if not matching_requests:
#                 continue

#             arrivals = sorted([r.arrival_time for r in matching_requests])
#             inter_arrival_times = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))] if len(arrivals) > 1 else [1.0]
#             dominated_request = max(matching_requests, key=lambda r: (NUM_REQUEST - r.num_completed_request) / r.qos_response_time)

#             num_replica = decide_provisioning(dominated_request, inter_arrival_times, ctn, edge_nodes)
#             for _ in range(num_replica):
#                 edge.add_replica(ctn["model"])

#         # Assignment phase
#         available_replicas = [
#             rep for node in edge_nodes for rep in node.replicas
#             if rep.busy_until <= max(r.arrival_time for r in request_sets)
#         ]
#         uncompleted_requests = [r for r in request_sets if r.num_completed_request < NUM_REQUEST]
#         assignments = self.assign_requests_to_replicas_with_hungarian(uncompleted_requests, available_replicas)

#         for rep_idx, req_idx in assignments:
#             replica = available_replicas[rep_idx]
#             req_set = uncompleted_requests[req_idx]
#             start, finish, accuracy = replica.process_request(req_set.arrival_time)

#             if round(finish, decimal_places(req_set.qos_response_time)) <= req_set.qos_response_time  and round(accuracy+sum(req_set.estimated_accuracy),max(decimal_places(accuracy), decimal_places(req_set.estimated_accuracy)))>=round(req_set.required_accuracy*(req_set.num_completed_request+1), decimal_places(req_set.required_accuracy)):
#                 completed_requests.append((req_set.arrival_time, start, finish))
#                 response_time = finish - req_set.arrival_time
#                 response_logs.append({
#                     "request_set_id": req_set.id,
#                     "arrival_time": req_set.arrival_time,
#                     "start_time": start,
#                     "finish_time": finish,
#                     "response_time": response_time,
#                     "model": replica.model.name,
#                     "edge_node_id": replica.edge_node.id
#                 })
#                 req_set.estimated_accuracy.append(accuracy)
#                 req_set.finish_time = max(req_set.finish_time, finish)
#                 req_set.num_completed_request += 1



class DecisionMaker_energy_request_priority:
    def __init__(self):
        pass

    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs):
        replicas_list = []
        total_model_flops = sum(m.required_flops for m in models)
        for node in edge_nodes:
            for model in models:
                model_weight = model.required_flops / total_model_flops
                flops = model_weight * node.total_flops_capacity * UTILIZATION_THRESHOLD
                energy = flops / node.flops_watt
                #Only proceed if node can allocate and has enough energy
                if node.can_allocate(flops) and node.available_energy >= energy:
                    replicas_list.append({
                        "edge_id": node.id,
                        "model": model,
                        "accuracy": model.accuracy,
                        "service_time": model.required_flops / flops,
                        "replica_flops": flops,
                        "energy": energy,
                        "current_replicas": len(node.replicas)
                    })

        replicas_list.sort(key=lambda r:r["energy"])
        for ctn in replicas_list:
            matching_requests = [
            r for r in request_sets
            if (NUM_REQUEST - r.num_completed_request) > 0 and
            round(sum(r.estimated_accuracy)+ctn["accuracy"], max(decimal_places(r.required_accuracy), decimal_places(ctn["accuracy"]))) >= round(r.required_accuracy*(r.num_completed_request+1), decimal_places(r.required_accuracy))] # request sets with some uncompleted requests

            if not matching_requests:
                edge_nodes[ctn["edge_id"]].remove_idle_replicas(
                    min(req.arrival_time for req in request_sets)
                )
                continue
            
            edge_nodes[ctn["edge_id"]].remove_idle_replicas(
                min(req.arrival_time for req in matching_requests)
            )
            
            matching_requests.sort(
                key=lambda r: (NUM_REQUEST - r.num_completed_request) / (r.qos_response_time + 1e-6),
                reverse=True
            )
            dominated_request = matching_requests[0]
            # dominated_request = max(matching_requests, key=lambda r: (NUM_REQUEST - r.num_completed_request) / r.qos_response_time)
            
            arrivals = sorted([r.arrival_time for r in matching_requests])
            inter_arrival_times = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))] if len(arrivals) > 1 else [1.0]

            num_replica = decide_provisioning(dominated_request, inter_arrival_times, ctn, edge_nodes)
            if num_replica <= 0:
                continue

            new_replicas = []
            for _ in range(num_replica):
                replica = edge_nodes[ctn["edge_id"]].add_replica(ctn["model"])
                if replica:
                    new_replicas.append(replica)

            for r_set in matching_requests:
                for replica in new_replicas:
                    for _ in range(NUM_REQUEST - r_set.num_completed_request):
                        start, finish, accuracy = replica.process_request(r_set.arrival_time)

                        if round(finish, decimal_places(r_set.qos_response_time)) <= round(r_set.arrival_time+r_set.qos_response_time,decimal_places(r_set.qos_response_time))  and round(accuracy+sum(r_set.estimated_accuracy),max(decimal_places(accuracy), decimal_places(r_set.estimated_accuracy)))>=round(r_set.required_accuracy*(r_set.num_completed_request+1), decimal_places(r_set.required_accuracy)):
                            
                            completed_requests.append((r_set.arrival_time, start, finish))
                            response_time = finish - r_set.arrival_time
                            response_logs.append({
                                "request_set_id": r_set.id,
                                "arrival_time": r_set.arrival_time,
                                "start_time": start,
                                "finish_time": finish,
                                "response_time": response_time,
                                "model": replica.model.name,
                                "edge_node_id": replica.edge_node.id
                            })
                            r_set.estimated_accuracy.append(accuracy)
                            r_set.finish_time = max(r_set.finish_time, finish)
                            r_set.num_completed_request += 1
                                
                
class DecisionMaker_edgeParams_request_priority:
    def __init__(self):
        pass

    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs):
        replicas_list = []
        total_model_flops = sum(m.required_flops for m in models)

        for node in edge_nodes:
            for model in models:
                model_weight = model.required_flops / total_model_flops
                flops = model_weight * node.total_flops_capacity * UTILIZATION_THRESHOLD
                energy = flops / node.flops_watt

                #Only proceed if node can allocate and has enough energy
                if node.can_allocate(flops) and node.available_energy >= energy:
                    replicas_list.append({
                        "edge_id": node.id,
                        "model": model,
                        "accuracy": model.accuracy,
                        "service_time": model.required_flops / flops,
                        "replica_flops": flops,
                        "energy": energy,
                        "current_replicas": len(node.replicas)
                    })

        # replicas_list.sort(key=lambda r: (r["energy"], r["service_time"], -r["accuracy"]))
        replicas_list.sort(key=lambda r: (r["service_time"], r["energy"], -r["accuracy"]))
        
        for ctn in replicas_list:
            matching_requests = [
            r for r in request_sets
            if (NUM_REQUEST - r.num_completed_request) > 0 and
            round(sum(r.estimated_accuracy)+ctn["accuracy"], max(decimal_places(r.required_accuracy), decimal_places(ctn["accuracy"]))) >= round(r.required_accuracy*(r.num_completed_request+1), decimal_places(r.required_accuracy))] # request sets with some uncompleted requests

            if not matching_requests:
                edge_nodes[ctn["edge_id"]].remove_idle_replicas(
                    min(req.arrival_time for req in request_sets)
                )
                continue

            matching_requests.sort(
                key=lambda r: (NUM_REQUEST - r.num_completed_request) / (r.qos_response_time + 1e-6),
                reverse=True
            )

            edge_nodes[ctn["edge_id"]].remove_idle_replicas(
                min(req.arrival_time for req in matching_requests)
            )

            dominated_request = matching_requests[0]
            # dominated_request = max(matching_requests, key=lambda r: (NUM_REQUEST - r.num_completed_request) / r.qos_response_time)
            arrivals = sorted([r.arrival_time for r in matching_requests])
            if len(arrivals)>=2:
                inter_arrival_times = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
            elif len(arrivals)==1:
                inter_arrival_times = arrivals

            num_replica = decide_provisioning(dominated_request, inter_arrival_times, ctn, edge_nodes)
            if num_replica <= 0:
                continue

            new_replicas = []
            for _ in range(num_replica):
                replica = edge_nodes[ctn["edge_id"]].add_replica(ctn["model"])
                if replica:
                    new_replicas.append(replica)

            for r_set in matching_requests:
                for replica in new_replicas:
                    for _ in range(NUM_REQUEST - r_set.num_completed_request):
                        start, finish, accuracy = replica.process_request(r_set.arrival_time)
                        if round(finish, decimal_places(r_set.qos_response_time)) <= round(r_set.arrival_time+r_set.qos_response_time,decimal_places(r_set.qos_response_time))  and round(accuracy+sum(r_set.estimated_accuracy),max(decimal_places(accuracy), decimal_places(r_set.estimated_accuracy)))>=round(r_set.required_accuracy*(r_set.num_completed_request+1), decimal_places(r_set.required_accuracy)):
                            
                            completed_requests.append((r_set.arrival_time, start, finish))
                            response_time = finish - r_set.arrival_time
                            response_logs.append({
                                "request_set_id": r_set.id,
                                "arrival_time": r_set.arrival_time,
                                "start_time": start,
                                "finish_time": finish,
                                "response_time": response_time,
                                "model": replica.model.name,
                                "edge_node_id": replica.edge_node.id
                            })
                            r_set.estimated_accuracy.append(accuracy)
                            r_set.finish_time = max(r_set.finish_time, finish)
                            r_set.num_completed_request += 1

class DecisionMaker_random:
    def __init__(self):
        pass
    
    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs):
        replicas_list = []
        max_attempts = sum(NUM_REQUEST-r.num_completed_request for r in request_sets)
        total_model_flops = sum(m.required_flops for m in models)

        attempts = 0
        while(attempts<max_attempts):
            req = random.choice(request_sets)
            if req.num_completed_request >= NUM_REQUEST:
                attempts += 1
                continue

            model = random.choice(models)
            node = random.choice(edge_nodes)

            replica = node.add_replica(model)

            if not replica:
                attempts += 1
                continue

            start, finish, acc = replica.process_request(req.arrival_time)
            if round(finish, decimal_places(req.qos_response_time)) <= req.qos_response_time  and round(acc+sum(req.estimated_accuracy),max(decimal_places(acc), decimal_places(req.estimated_accuracy)))>=round(req.required_accuracy*(req.num_completed_request+1), decimal_places(req.required_accuracy)):
                completed_requests.append((req.arrival_time, start, finish))
                response_time = finish - req.arrival_time
                response_logs.append({
                    "request_set_id": req.id,
                    "arrival_time": req.arrival_time,
                    "start_time": start,
                    "finish_time": finish,
                    "response_time": response_time,
                    "model": replica.model.name,
                    "edge_node_id": replica.edge_node.id
                })
                req.estimated_accuracy.append(acc)
                req.finish_time = max(req.finish_time, finish)
                req.num_completed_request += 1

            attempts += 1



def decide_provisioning(dominated_request, inter_arrival_times, ctn, edge_nodes):
        
        edge = edge_nodes[ctn["edge_id"]]
        max_replicas = min(math.floor((edge.total_flops_capacity-edge.used_flops())/ctn["replica_flops"]), 
                           math.floor((edge.total_energy_limit - edge.used_energy())/ctn["energy"])) #maximum number of acceptable replicas
        
        service_time = ctn["service_time"]
        lambda_rate = 1 / np.mean(inter_arrival_times)
        mu = 1 / service_time
        var_inter = np.var(inter_arrival_times) if len(inter_arrival_times) > 1 else 0
        ca2 = var_inter / (np.mean(inter_arrival_times))
        cs2 = mu
        
        wait_time = float('inf') 
        req = 0
        num_replica = sum(1 for rep in edge_nodes[ctn["edge_id"]].replicas if rep.model.name == ctn["model"].name)
        
        for num_replica in range(1,max_replicas+1):
            rho = lambda_rate / (num_replica * mu)
            if rho >= 1:
                continue
            wq = ((ca2**2 + cs2**2) / 2) * (rho ** (np.sqrt(2 * (num_replica + 1)) - 1)) / (num_replica * (1 - rho)) / mu 
            wait_time = wq + 1 / mu
            req = min(num_replica, NUM_REQUEST-dominated_request.num_completed_request)
            
            if round(wait_time+ctn["service_time"], max(decimal_places(wait_time), decimal_places(ctn["service_time"])))<=dominated_request.qos_response_time and req+dominated_request.num_completed_request==NUM_REQUEST and (lambda_rate/((num_replica+1)*mu))<=UTILIZATION_THRESHOLD: 
                return num_replica
    
        # print("num_replica:  ", num_replica)    
        return num_replica
  

class Simulator:
    def __init__(self, edge_nodes, models, duration, decision_maker):
        self.edge_nodes = edge_nodes
        self.models = models
        self.duration = duration

        self.request_sets = []
        self.completed_requests = []
        self.decision_maker = decision_maker
        self.decision_times = []
        self.response_logs = []


    def run(self):
        start_sim_time = 0
        decision_times = []
        maximum_arrivals = max(st.arrival_time for st in self.request_sets)
        while(start_sim_time<=maximum_arrivals):
            
            current_requests = []
            current_requests = [st for st in self.request_sets if st.arrival_time>start_sim_time and st.arrival_time<start_sim_time+TIME_SLOT]

            
            if current_requests:
                for st in current_requests:
                    st.arrival_time=start_sim_time+TIME_SLOT
                    st.flag = True  
                start_decision = time.time()
                self.decision_maker.decide_allocation(current_requests, self.edge_nodes, self.models, self.completed_requests, self.response_logs)
                end_decision = time.time()
                decision_times.append((end_decision - start_decision))

            start_sim_time += TIME_SLOT
        self.decision_times = decision_times


    def plot_metrics(self, delays,  diff_time, diff_avg_acc, approach):

        # Comparison of qos_response_time vs finish_time
        qos_times = [req.qos_response_time for req in self.request_sets]
        finish_times = [req.finish_time for req in self.request_sets]
        indices1 = np.arange(len(self.request_sets))
    
        plt.figure(figsize=(12, 6))
        width = 0.35
        plt.bar(indices1 - width/2, qos_times, width=width, label='Deadline', color='skyblue')
        plt.bar(indices1 + width/2, finish_times, width=width, label='Response Time', color='salmon')
        plt.xlabel("Request Set Index")
        plt.ylabel("Time (seconds)")
        plt.title(f"QoS Deadline vs Actual Finish Time per Request Set in approach {approach}")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Comparison of estimated avg accuracy vs qos_accuarcy
        qos_accuracy = [req.required_accuracy for req in self.request_sets if req.num_completed_request>0]
        avg_accuracy = [(sum(req.estimated_accuracy)/req.num_completed_request) for req in self.request_sets if req.num_completed_request>0]
        if qos_accuracy:
            indices2 = np.arange(len([req for req in self.request_sets if req.num_completed_request>0]))
            plt.figure(figsize=(12, 6))
            width = 0.35
            plt.bar(indices2 - width/2, qos_accuracy, width=width, label='Expected accuracy', color='skyblue')
            plt.bar(indices2 + width/2, avg_accuracy, width=width, label='Estimated avg accuracy', color='salmon')
            plt.xlabel("Request Set Index")
            plt.ylabel("Accuracy")
            plt.title(f"QoS accuracy vs estimated avg accuracy for Request Sets with completed requests in approach {approach}")
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

        
            

    def print_stats(self, approach, edge_num, arrival, energy_interval, log):
        print(f"---------------------------  Approach:  {approach}  ---------------------------------")
        total = NUM_REQUEST*len(self.request_sets)
        # print("first request set:  ", self.request_sets[0])
        completed = sum([req.num_completed_request for req in self.request_sets])
        rejected = sum([NUM_REQUEST-req.num_completed_request for req in self.request_sets if req.flag==True])
        delays =  [finish - arrival for arrival, _, finish in self.completed_requests]
        diff_time = [(req.qos_response_time - req.finish_time) for req in self.request_sets]
        diff_avg_acc = [(sum(req.estimated_accuracy)/req.num_completed_request)- req.required_accuracy for req in self.request_sets if req.num_completed_request>0] #TODO : /devide number
        total_diff_time = sum([((req.arrival_time+req.qos_response_time) - req.finish_time) for req in self.request_sets  if req.num_completed_request>0])
        total_diff_avg_acc = sum([(sum(req.estimated_accuracy)/req.num_completed_request)- req.required_accuracy for req in self.request_sets if req.num_completed_request>0])

        print(f"Total Requests: {total}")
        print(f"Completed Requests: {(completed)}")
        print(f"Rejected Requests: {(rejected)}")
        print(f"Difference between finish time of completed request and deadline, for all request sets:  ", total_diff_time)
        print(f"Difference between average accuracy of completed request and expected minimum accuracy, for request sets with completed requests:  ", total_diff_avg_acc)

        if (completed) > 0:
            avg_delay = sum(delays) / total
            percentile_90 = np.percentile(delays, 90)
            percentile_95 = np.percentile(delays, 95)
            print(f"Average Total Time per Request: {avg_delay:.2f} seconds")
            print(f"90th Percentile Response Time: {percentile_90:.2f} seconds")
            print(f"95th Percentile Response Time: {percentile_95:.2f} seconds")

        total_energy_used = sum(node.used_energy() for node in self.edge_nodes)
        print(f"Total Energy Consumed: {total_energy_used:.2f}")

        total_util = 0
        for node in self.edge_nodes:
            used_flops = node.used_flops()
            print(f"node: {node.id} used this much flops: {used_flops}")
            print(f"node: {node.id} used this much energy: {node.used_energy()}")
            flops_util = used_flops / (UTILIZATION_THRESHOLD * node.total_flops_capacity)
            print(f"Node {node.id} FLOPs Utilization: {flops_util:.2%}")
            total_util += flops_util
        total_util/=len(self.edge_nodes)

        if hasattr(self, 'decision_times'):
            avg_decision_time = sum(self.decision_times) / len(self.decision_times)
            perc_90_decision = np.percentile(self.decision_times, 90)
            print(f"Average Decision Time per Slot: {avg_decision_time:.4f} seconds")
            print(f"90th Percentile Decision Time: {perc_90_decision:.4f} seconds")

        if log is not None:

            if energy_interval is None:
                energy_interval= [10**4, 10**6] # Defaulf interval
    
            log.append({
                        "approach": approach,
                        "edge_num": edge_num, 
                        "Arrival_rate": arrival,
                        "Energy_interval": energy_interval,
                        "total_request": total,
                        "Completed_requests": completed,
                        "Rejected_requests": rejected,
                        "Speed:Deadline-Finish": total_diff_time,
                        "Acc:AvgAcc-ExpAcc": total_diff_avg_acc,
                        "Total_energy_used": total_energy_used,
                        "Total_utilisation": total_util 
                    })    
        # self.plot_metrics(delays, diff_time, diff_avg_acc, approach)

    def save_response_logs(self, approach):
        filename=f"{approach}_response_logs.csv"
        if not self.response_logs:
            print("No response logs to save.")
            return

        with open(filename, mode='w', newline='') as csvfile:
            fieldnames = ["request_set_id", "arrival_time", "start_time", "finish_time", "response_time", "model",
                          "edge_node_id"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.response_logs:
                writer.writerow(entry)

        print(f"Saved response logs to {filename}")
    


def decimal_places(val):
    s = str(val)
    if '.' in s:
        return len(s.split('.')[-1])
    return 0


    
def main(Experiments):
    
    DEFAULT_EDGE_NUM = 5
    DEFAULT_ENERGY_INTERVAL = [10**3, 10**4]
    DEFAULT_ARRIVAL = 5
    NUM_REQUEST = 10

    random.seed(42)
    np.random.seed(42)

    energy_scale = [random.randint(2,8) for i in range(DEFAULT_EDGE_NUM)]    

    models = [
        MLModel(name="yolov5n", required_flops=7.7, accuracy= 0.3),
        MLModel(name="yolov5s", required_flops=24, accuracy= 0.4),
        MLModel(name="yolov5m", required_flops=64.2, accuracy= 0.58),
        MLModel(name="yolov5l", required_flops=135.0, accuracy= 0.88),
        MLModel(name="yolov5x", required_flops=246.4, accuracy= 0.95)
    ]

    def build_nodes(edge_num, scale_param):
        if scale_param is None:
            edge_nodes = []
            for i in range(edge_num):
                print("energy_limit:  ", (WATTS[i%DEFAULT_EDGE_NUM]*(3600)*energy_scale[i%DEFAULT_EDGE_NUM]) )
                node = EdgeNode(id=i,
                         flops_capacity=FLOPS[i%DEFAULT_EDGE_NUM],
                         flops_watt=FLOPS_WATT[i%DEFAULT_EDGE_NUM],
                         energy_limit = (WATTS[i%DEFAULT_EDGE_NUM]*(3600)*energy_scale[i%DEFAULT_EDGE_NUM]),
                         models=copy.deepcopy(models))
                edge_nodes.append(node)
            return edge_nodes

        else:
            edge_nodes = []
            for i in range(edge_num):
                print("energy limit:  ", (WATTS[i%DEFAULT_EDGE_NUM]*(3600)*energy_scale[i%DEFAULT_EDGE_NUM])**scale_param)
                node = EdgeNode(id=i,
                         flops_capacity=FLOPS[i%DEFAULT_EDGE_NUM],
                         flops_watt=FLOPS_WATT[i%DEFAULT_EDGE_NUM],
                         energy_limit = (WATTS[i%DEFAULT_EDGE_NUM]*(3600)*energy_scale[i%DEFAULT_EDGE_NUM])**1,
                         models=models)
                edge_nodes.append(node)
            return edge_nodes

    def generate_poisson_events(rate):
        num_events = np.random.poisson(rate)
        event_times = np.sort(np.random.uniform(0, 60, num_events))
        return event_times
            


    def generate_requestSet(arrival):
        
        arrivals = set()
        base_requests = []
        if arrival is None:
            arrival = DEFAULT_ARRIVAL
            
        poission_requests = generate_poisson_events(50000)
        with open("time_stamps_alibaba.csv", newline='') as file:
            alibaba_traces = csv.reader(file)
    
            req_id = 0
            ii = 0
            for idx, row in enumerate(islice(alibaba_traces, 0, 50000, arrival)):
                try:
                    row = [int(item.strip()) for item in row]
                except ValueError:
                    continue
    
                # arrivals.add(row[0])
                required_accuracy = round(random.uniform(MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY), 2) 
                qos_response_time = round(random.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME), 2)  
                base_requests.append(Request_set(req_id, row[0], required_accuracy, qos_response_time, 0, [], 0, False))
                req_id +=1 

                # required_accuracy = round(random.uniform(MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY), 2) 
                # qos_response_time = round(random.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME), 2)  
                # base_requests.append(Request_set(req_id, poission_requests[ii], required_accuracy, qos_response_time, 0, [], 0, False))
                # req_id +=1
                # ii +=1 

        
        return base_requests         

    base_requests = generate_requestSet(None)
    if Experiments is False:
        edge_num = DEFAULT_EDGE_NUM
        sim1 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker= DecisionMaker_energy_priority())
        sim1.request_sets = copy.deepcopy(base_requests)
        sim1.run()
        approach = "Energy-aware"
        sim1.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, None)
        sim1.save_response_logs(approach)
            
        sim2 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_energy_request_priority())
        sim2.request_sets = copy.deepcopy(base_requests)
        sim2.run()
        approach = "Deadline-energy-aware"
        sim2.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, None)
        sim2.save_response_logs(approach)

        sim3 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_edgeParams_request_priority())
        sim3.request_sets = copy.deepcopy(base_requests)
        sim3.run()
        approach = "Multi-criteria"
        sim3.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, None)
        sim3.save_response_logs(approach)
                    
        # sim4 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker=DecisionMaker_hungarian())
        # sim4.request_sets = copy.deepcopy(base_requests)
        # sim4.run()
        # approach = "hungarian"
        # sim4.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, None)
        # sim4.save_response_logs(approach)
        
        # sim5 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_random())
        # sim5.request_sets = copy.deepcopy(base_requests)
        # sim5.run()
        # approach = "Random"
        # sim4.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, None)
        # sim4.save_response_logs(approach)

    else:
        log_diffEdge = []
        # base_requests = generate_requestSet(None)
        for edge_num in range(2,20):
            print("edge_num:  ", edge_num)
            sim1 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker=DecisionMaker_energy_priority())
            sim1.request_sets = copy.deepcopy(base_requests)
            sim1.run()
            approach = "Energy-aware"
            sim1.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, log_diffEdge)
            sim1.save_response_logs(approach)
            
            sim2 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_energy_request_priority())
            sim2.request_sets = copy.deepcopy(base_requests)
            sim2.run()
            approach = "Deadline-energy-aware"
            sim2.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, log_diffEdge)
            sim2.save_response_logs(approach)

            sim3 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_edgeParams_request_priority())
            sim3.request_sets = copy.deepcopy(base_requests)
            sim3.run()
            approach = "Multi-criteria"
            sim3.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, log_diffEdge)
            sim3.save_response_logs(approach)
                    
            # sim4 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker=DecisionMaker_hungarian())
            # sim4.request_sets = copy.deepcopy(base_requests)
            # sim4.run()
            # approach = "Hungarian"
            # sim4.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, log_diffEdge)
            # sim4.save_response_logs(approach)
        
            # sim5 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_random())
            # sim5.request_sets = copy.deepcopy(base_requests)
            # sim5.run()
            # approach = "Random"
            # sim5.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, log_diffEdge)
    
        pd.DataFrame(log_diffEdge).to_csv("log_diffEdge.csv", index=False)
        edge_num = DEFAULT_EDGE_NUM  # Number of edge servers by default
        # base_requests = generate_requestSet(None)
        log_diffEnergy = []
        scal_param = [
            (1/50),
            (1/30),
            (1/200),
            (2/100),
            (1)
        ]
        
        for param in scal_param:
            print("param:  ", param)
            sim1 = Simulator(edge_nodes=build_nodes(edge_num, param), models=models, duration=60, decision_maker=DecisionMaker_energy_priority())
            sim1.request_sets = copy.deepcopy(base_requests)
            sim1.run()
            approach = "Energy-aware"
            sim1.print_stats(approach, edge_num, DEFAULT_ARRIVAL, param, log_diffEnergy)

            
            sim2 = Simulator(edge_nodes=build_nodes(edge_num, param), models=models, duration=60, decision_maker = DecisionMaker_energy_request_priority())
            sim2.request_sets = copy.deepcopy(base_requests)
            sim2.run()
            approach = "Deadline-energy-aware"
            sim2.print_stats(approach, edge_num, DEFAULT_ARRIVAL, param, log_diffEnergy)

            sim5 = Simulator(edge_nodes=build_nodes(edge_num, param), models=models, duration=60, decision_maker = DecisionMaker_edgeParams_request_priority())
            sim5.request_sets = copy.deepcopy(base_requests)
            sim5.run()
            approach = "Multi-criteria"
            sim5.print_stats(approach, edge_num, DEFAULT_ARRIVAL, param, log_diffEnergy)

                    
        #     sim3 = Simulator(edge_nodes=build_nodes(edge_num, param), models=models, duration=60, decision_maker=DecisionMaker_hungarian())
        #     sim3.request_sets = copy.deepcopy(base_requests)
        #     sim3.run()
        #     approach = "Hungarian"
        #     sim3.print_stats(approach, edge_num, DEFAULT_ARRIVAL, param, log_diffEnergy)

        
        #     sim4 = Simulator(edge_nodes=build_nodes(edge_num, param), models=models, duration=60, decision_maker = DecisionMaker_random())
        #     sim4.request_sets = copy.deepcopy(base_requests)
        #     sim4.run()
        #     approach = "Random"
        #     sim4.print_stats(approach, edge_num, DEFAULT_ARRIVAL, param, log_diffEnergy)
            
    
        pd.DataFrame(log_diffEnergy).to_csv("log_diffEnergy.csv", index=False)
        
        log_diffArrv = []
    
        edge_num = DEFAULT_EDGE_NUM  # Number of edge servers by default
        diff_arrivals = [50,40,30,20,10,5]
        for arrv in diff_arrivals:
            base_requests = generate_requestSet(arrv)
            sim1 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker=DecisionMaker_energy_priority())
            sim1.request_sets = copy.deepcopy(base_requests)
            sim1.run()
            approach = "Energy-aware"
            sim1.print_stats(approach, edge_num, arrv, None, log_diffArrv)
            sim1.save_response_logs(approach)
            
            sim2 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_energy_request_priority())
            sim2.request_sets = copy.deepcopy(base_requests)
            sim2.run()
            approach = "Deadline-energy-aware"
            sim2.print_stats(approach, edge_num, arrv, None, log_diffArrv)
            sim2.save_response_logs(approach)


            sim5 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_edgeParams_request_priority())
            sim5.request_sets = copy.deepcopy(base_requests)
            sim5.run()
            approach = "Multi-criteria"
            sim5.print_stats(approach, edge_num, arrv, None, log_diffArrv)
            sim5.save_response_logs(approach)
                    
        #     sim3 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker=DecisionMaker_hungarian())
        #     sim3.request_sets = copy.deepcopy(base_requests)
        #     sim3.run()
        #     approach = "Hungarian"
        #     sim3.print_stats(approach, edge_num, arrv, None, log_diffArrv)
        #     sim3.save_response_logs(approach)
        
        #     sim4 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_random())
        #     sim4.request_sets = copy.deepcopy(base_requests)
        #     sim4.run()
        #     approach = "Random"
        #     sim4.print_stats(approach, edge_num, arrv, None, log_diffArrv)
    
        pd.DataFrame(log_diffArrv).to_csv("log_diffArrv.csv", index=False)

    
    

if __name__ == "__main__":
    main()
