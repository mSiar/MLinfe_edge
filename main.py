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
        self.accuracy = accuracy 


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
         return self.model.required_flops/self.edge_node.flops_watt


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

        return flops_needed <= self.available_flops

    def current_energy_usage(self):
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
        estimated_energy = model.required_flops/self.flops_watt
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
        self.replicas = active_replicas

class Request_set:
    def __init__(self, id, arrival_time, qos_accuracy, qos_response_time, num_completed_request, estimated_accuracy, finish_time, flag):  
        self.id = id
        self.arrival_time = arrival_time
        self.num_completed_request = num_completed_request  #Initially zero
        self.estimated_accuracy = estimated_accuracy
        self.finish_time = finish_time
        self.required_accuracy = qos_accuracy
        self.qos_response_time = qos_response_time
        self.flag = False


class DecisionMaker_energy_aware:
    def __init__(self):
        pass
    
    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs):
        replicas_list = []
        total_model_flops = sum(m.required_flops for m in models)
        flag = False
        cc = 0
        for node in edge_nodes:
            for model in models:
                model_weight = model.required_flops / total_model_flops
                flops = model_weight * node.total_flops_capacity * UTILIZATION_THRESHOLD
                energy = model.required_flops / node.flops_watt
                if node.can_allocate(flops) and node.available_energy >= energy:
                    replicas_list.append({"edge_id": node.id,      
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
                edge_nodes[ctn["edge_id"]].remove_idle_replicas(min(req.arrival_time for req in request_sets)- 1e-6 ) #remove_idle_replicas
                continue

            matching_requests.sort(key=lambda r:r.arrival_time)
            edge_nodes[ctn["edge_id"]].remove_idle_replicas(min(req.arrival_time for req in matching_requests)- 1e-6)   #remove_idle_replicas
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
                                cc+=1
                


class DecisionMaker_deadline_energy_aware:
    def __init__(self):
        pass

    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs):
        replicas_list = []
        total_model_flops = sum(m.required_flops for m in models)
        for node in edge_nodes:
            for model in models:
                model_weight = model.required_flops / total_model_flops
                flops = model_weight * node.total_flops_capacity * UTILIZATION_THRESHOLD
                energy = model.required_flops / node.flops_watt
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
                    min(req.arrival_time for req in request_sets)-1e-6
                )
                continue
            
            edge_nodes[ctn["edge_id"]].remove_idle_replicas(
                min(req.arrival_time for req in matching_requests)-1e-6
            )
            
            matching_requests.sort(
                key=lambda r: (NUM_REQUEST - r.num_completed_request) / (r.qos_response_time + 1e-6),
                reverse=True
            )
            dominated_request = matching_requests[0]
            
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

                                
                
class DecisionMaker_multi_criteria:
    def __init__(self):
        pass

    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs):
        replicas_list = []
        total_model_flops = sum(m.required_flops for m in models)

        for node in edge_nodes:
            for model in models:
                
                model_weight = model.required_flops / total_model_flops
                flops = model_weight * node.total_flops_capacity * UTILIZATION_THRESHOLD
                energy = model.required_flops / node.flops_watt

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

        replicas_list.sort(key=lambda r: (r["service_time"], r["energy"], -r["accuracy"]))
        
        for ctn in replicas_list:
            matching_requests = [
            r for r in request_sets
            if (NUM_REQUEST - r.num_completed_request) > 0 and
            round(sum(r.estimated_accuracy)+ctn["accuracy"], max(decimal_places(r.required_accuracy), decimal_places(ctn["accuracy"]))) >= round(r.required_accuracy*(r.num_completed_request+1), decimal_places(r.required_accuracy))] # request sets with some uncompleted requests

            if not matching_requests:
                edge_nodes[ctn["edge_id"]].remove_idle_replicas(
                    min(req.arrival_time for req in request_sets)-1e-6
                )
                continue

            matching_requests.sort(
                key=lambda r: (NUM_REQUEST - r.num_completed_request) / (r.qos_response_time + 1e-6),
                reverse=True
            )

            edge_nodes[ctn["edge_id"]].remove_idle_replicas(
                min(req.arrival_time for req in matching_requests)-1e-6
            )

            dominated_request = matching_requests[0]
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
                decision_times.append(end_decision - start_decision)

            start_sim_time += TIME_SLOT
        self.decision_times = decision_times        
            

    def print_stats(self, approach, edge_num, arrival_rate, log):
        print(f"---------------------------  Approach:  {approach}  ---------------------------------")
        total = NUM_REQUEST*len(self.request_sets)
        completed = sum([req.num_completed_request for req in self.request_sets])
        rejected = sum([NUM_REQUEST-req.num_completed_request for req in self.request_sets if req.flag==True])
        delays =  [finish - arrival for arrival, _, finish in self.completed_requests]
        diff_time = [(req.qos_response_time - req.finish_time) for req in self.request_sets]
        diff_avg_acc = [(sum(req.estimated_accuracy)/req.num_completed_request)- req.required_accuracy for req in self.request_sets if req.num_completed_request>0] 
        total_diff_time = sum([((req.arrival_time+req.qos_response_time) - req.finish_time) for req in self.request_sets  if req.num_completed_request>0])
        total_diff_avg_acc = sum([(sum(req.estimated_accuracy)/req.num_completed_request)- req.required_accuracy for req in self.request_sets if req.num_completed_request>0])

        print(f"Total Requests: {total}")
        print(f"Completed Requests: {(completed)}")
        print(f"Rejected Requests: {(rejected)}")
        print(f"Difference between finish time of completed request and deadline, for all request sets:  ", total_diff_time)
        print(f"Difference between average accuracy of completed request and expected minimum accuracy, for request sets with completed requests:  ", total_diff_avg_acc)


        total_energy_used = sum(node.used_energy() for node in self.edge_nodes)
        print(f"Total Energy Consumed: {total_energy_used:.2f}")

        for node in self.edge_nodes:
            print(f"node: {node.id} used this much energy: {node.used_energy()}")
            print(f"node: {node.id} energy limit:  {node.total_energy_limit}")

        if hasattr(self, 'decision_times'):
            avg_decision_time = sum(self.decision_times) / len(self.decision_times)
            perc_90_decision = np.percentile(self.decision_times, 90)
            print(f"Average Decision Time per Slot: {avg_decision_time:.4f} seconds")

        if log is not None:
            log.append({
                        "approach": approach,
                        "edge_num": edge_num, 
                        "Arrival_rate": arrival_rate,
                        "total_request": total,
                        "Completed_requests": completed,
                        "Rejected_requests": rejected,
                        "Speed:Deadline-Finish": total_diff_time,
                        "Acc:AvgAcc-ExpAcc": total_diff_avg_acc,
                        "Total_energy_used": total_energy_used,
                    })    

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


    
def main():
    
    DEFAULT_EDGE_NUM = 5
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
                node = EdgeNode(id=i,
                         flops_capacity=FLOPS[i%DEFAULT_EDGE_NUM],
                         flops_watt=FLOPS_WATT[i%DEFAULT_EDGE_NUM],
                         energy_limit = (WATTS[i%DEFAULT_EDGE_NUM]*(3600)*energy_scale[i%DEFAULT_EDGE_NUM])**scale_param,
                         models=copy.deepcopy(models))
                edge_nodes.append(node)
            return edge_nodes

    
    def generate_poisson_events(rate):
        num_events = np.random.poisson(rate)
        event_times = np.sort(np.random.uniform(0, 60, num_events))
        return event_times
            
    poission_requests = generate_poisson_events(200)
    
    def generate_requestSet(arrival):
        
        base_requests = []
        if arrival is None:
            arrival = DEFAULT_ARRIVAL
        with open("time_stamps_alibaba.csv", newline='') as file:
            alibaba_traces = csv.reader(file)
            
            req_id = 0
            ii = 0
            for idx, row in enumerate(islice(alibaba_traces, 0, 25000, arrival)):
                try:
                    row = [int(item.strip()) for item in row]
                except ValueError:
                    continue
                required_accuracy = round(random.uniform(MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY), 2) 
                qos_response_time = round(random.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME), 2)  
                base_requests.append(Request_set(req_id, row[0], required_accuracy, qos_response_time, 0, [], 0, False))
                req_id +=1 

        for p_time  in poission_requests[::arrival]:
                required_accuracy = round(random.uniform(MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY), 2) 
                qos_response_time = round(random.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME), 2)  
                base_requests.append(Request_set(req_id, p_time, required_accuracy, qos_response_time, 0, [], 0, False))
                req_id +=1

        
        return base_requests         

    base_requests = generate_requestSet(None)

    edge_num = DEFAULT_EDGE_NUM        
    
    sim1 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker= DecisionMaker_energy_aware())
    sim1.request_sets = copy.deepcopy(base_requests)
    sim1.run()
    approach = "Energy-aware"
    sim1.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None)
    sim1.save_response_logs(approach)

    
    sim2 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_deadline_energy_aware())
    sim2.request_sets = copy.deepcopy(base_requests)
    sim2.run()
    approach = "Deadline-energy-aware"
    sim2.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None)
    sim2.save_response_logs(approach)

    
    sim3 = Simulator(edge_nodes=build_nodes(edge_num, None), models=models, duration=60, decision_maker = DecisionMaker_multi_criteria())
    sim3.request_sets = copy.deepcopy(base_requests)
    sim3.run()
    approach = "Multi-criteria"
    sim3.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None)
    sim3.save_response_logs(approach)

    
    

if __name__ == "__main__":
    main()
