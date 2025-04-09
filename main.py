import random
import numpy as np
import math
import matplotlib.pyplot as plt
from setting import NUM_REQUEST, UTILIZATION_THRESHOLD, MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY, MIN_RESPONSE_TIME, MAX_RESPONSE_TIME, TIME_SLOT, FLOPS, FLOPS_WATT


class MLModel:
    def __init__(self, name, required_flops, accuracy):
        self.name = name
        self.required_flops = required_flops
        self.accuracy = accuracy #this not dependent on allocated FLOPs, hence static


class Replica:
    # DONE: Change it according to the relation of flops and energy consumption per flop equation

    def __init__(self, allocated_flops, model: MLModel, edge_node):
        self.edge_node = edge_node
        self.model = model
        self.allocated_flops = allocated_flops  #(model.required_flops/(sum ml.required_flops for ml in MLModel))*
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
         return self.allocated_flops*self.edge_node.flops_watt


class EdgeNode:
    def __init__(self, id, flops_capacity, flops_watt, energy_limit, models):
        self.id = id
        self.total_flops_capacity = flops_capacity
        self.flops_watt = flops_watt
        self.total_energy_limit = energy_limit
        self.available_flops = flops_capacity*UTILIZATION_THRESHOLD   ## CHECK: DO WE NEED multiplying by UTILIZATION_THRESHOLD ??
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
        return used_flops  ## CHECK: AM I NEED UTILIZATION_THRESHOLD ??

    def used_energy(self):
        return self.total_energy_limit - self.available_energy

    def add_replica(self, model): 
        total_model_flops = sum(m.required_flops for m in self.models)
        model_weight = model.required_flops / total_model_flops
        flops = model_weight*self.total_flops_capacity*UTILIZATION_THRESHOLD ## CHECK: AM I NEED UTILIZATION_THRESHOLD ??
        estimated_energy = flops* self.flops_watt
        if self.can_allocate(flops) and self.available_energy >= estimated_energy:
            replica = Replica(flops, model, self)
            self.replicas.append(replica)
            self.available_flops -= flops
            self.available_energy -= estimated_energy

            return replica

        return None

    def remove_idle_replicas(self, current_time): #####?????? HOW 
        active_replicas = []
        for rep in self.replicas:
            if rep.busy_until > current_time:
                active_replicas.append(rep)
            else:
                self.available_flops += rep.allocated_flops
                self.available_energy += rep.energy_consumption()
        self.replicas = active_replicas

class Request_set:
    def __init__(self, arrival_time, qos_accuracy, qos_response_time, num_completed_request, estimated_accuracy, finish_time):  #num_request is the number of assigned requests in the set,       estimated_accuracy is the accuracy of the ---> initialized []
        self.arrival_time = arrival_time
        self.num_completed_request = num_completed_request  #Initially zero
        self.estimated_accuracy = estimated_accuracy
        self.finish_time = finish_time
        self.required_accuracy = qos_accuracy
        self.qos_response_time = qos_response_time


class DecisionMaker:
    def __init__(self):
        pass

    def decide_provisioning(self, dominated_request, inter_arrival_times, ctn, edge_nodes):
        
        max_replicas = min(math.floor((edge_nodes[ctn["edge_id"]].total_flops_capacity-edge_nodes[ctn["edge_id"]].used_flops())/ctn["replica_flops"]), math.floor((edge_nodes[ctn["edge_id"]].total_energy_limit-edge_nodes[ctn["edge_id"]].used_energy())/ctn["energy"])) #maximum number of acceptable replicas
        
        service_time = ctn["service_time"]
        lambda_rate = 1 / np.mean(inter_arrival_times)
        mu = 1 / service_time
        if len(inter_arrival_times)>1:
            var_inter = np.var(inter_arrival_times)
        else:
            var_inter =0 
        ca2 = var_inter / (np.mean(inter_arrival_times))
        cs2 = mu
        
        wait_time = float('inf') 
        req = 0
        num_replica = sum(1 for rep in edge_nodes[ctn["edge_id"]].replicas if rep.model.name == ctn["model"].name)
        
        for num_replica in range(1,max_replicas+1):
            rho = lambda_rate / (num_replica * mu)
            if rho >= 1:
                continue
            wq = ((ca2**2 + cs2**2) / 2) * (rho ** (np.sqrt(2 * (num_replica + 1)) - 1)) / (num_replica * (1 - rho)) / mu # Formula changed
            wait_time = wq + 1 / mu
            req = min(math.floor((wait_time+ctn["service_time"])/num_replica), NUM_REQUEST-dominated_request.num_completed_request)
            if(wait_time+ctn["service_time"]<=dominated_request.qos_response_time and num_replica <=max_replicas and req+dominated_request.num_completed_request<NUM_REQUEST):  ##?? CHECK:  dont we need UTILIZATION_THRESHOLD here??
                return num_replica
    
        return num_replica
    

    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests):#, rejected_requests):
        replicas_list = []
        total_model_flops = sum(m.required_flops for m in models)
        
        for node in edge_nodes:
            for model in models:
                model_weight = model.required_flops / total_model_flops
                flops = model_weight * node.total_flops_capacity* UTILIZATION_THRESHOLD #CHECK: UTILIZATION_THRESHOLD ?
                energy = flops * node.flops_watt

                replicas_list.append({"edge_id": node.id,       #List of all potential containers CTN_j,l the system
                                      "model": model,
                                      "accuracy": model.accuracy,
                                      "service_time": model.required_flops/flops,
                                      "replica_flops":flops,
                                      "energy": energy,
                                      "current_replicas": len(node.replicas)
                                     })
        replicas_list.sort(key=lambda r:r["energy"])        

        for ctn in replicas_list:  
            matching_requests = [r for r in request_sets if (NUM_REQUEST-r.num_completed_request)>0 and sum(r.estimated_accuracy)+ctn["accuracy"] >= (r.num_completed_request+1)*r.required_accuracy] # request sets with some uncompleted requests
            
            if not matching_requests:
                edge_nodes[ctn["edge_id"]].remove_idle_replicas(max(req.arrival_time for req in request_sets) ) #just removing idle replicas from container
            else:
                edge_nodes[ctn["edge_id"]].remove_idle_replicas(max(req.arrival_time for req in matching_requests))   #remove_idle_replicas
                dominated_request = max(matching_requests,key=lambda r: (NUM_REQUEST - r.num_completed_request) / r.qos_response_time) # The set with maximum number of uncompleted requests and minimum deadline
    
                arrivals = [r.arrival_time for r in matching_requests]
                if len(arrivals)>=2:
                    inter_arrival_times = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
                elif len(arrivals)==1:
                    inter_arrival_times = arrivals
                num_replica = self.decide_provisioning(dominated_request, inter_arrival_times, ctn, edge_nodes)  # Provisioning decision
                
                if num_replica>0:
                    matching_requests.sort(key=lambda r:r.arrival_time)
            
                    for _ in range(num_replica):
                        edge_nodes[ctn["edge_id"]].add_replica(ctn["model"])
                    
                    for r_set in matching_requests:
                        for replica in edge_nodes[ctn["edge_id"]].replicas:
                            for req in range(NUM_REQUEST-r_set.num_completed_request):
                                start, finish, accuracy = replica.process_request(r_set.arrival_time)
                                if finish <= r_set.qos_response_time and accuracy+sum(r_set.estimated_accuracy)>=r_set.required_accuracy*(r_set.num_completed_request+1):
                                    completed_requests.append((r_set.arrival_time, start, finish))
                                    r_set.estimated_accuracy.append(accuracy)
                                    r_set.finish_time = max(r_set.finish_time, finish)
                                    r_set.num_completed_request +=1


class Simulator:
    def __init__(self, edge_nodes, models, arrival_rate_fn, duration, decision_maker):
        self.edge_nodes = edge_nodes
        self.models = models
        self.arrival_rate_fn = arrival_rate_fn
        self.duration = duration

        self.request_sets = []
        self.completed_requests = []
        self.decision_maker = decision_maker

    
    def generate_requests(self):
        current_time = 0
        while current_time < self.duration:
            inter_arrival = self.arrival_rate_fn()
            current_time += inter_arrival
            required_accuracy = round(random.uniform(MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY),2)   
            qos_response_time = current_time+round(random.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME),2)  
            self.request_sets.append(Request_set(current_time, required_accuracy, qos_response_time, 0, [], 0))


    def run(self):
        start_sim_time = 0
        self.generate_requests()
        maximum_arrivals = max(st.arrival_time for st in self.request_sets)
        while(start_sim_time<maximum_arrivals):
            current_requests = [st for st in self.request_sets if st.arrival_time>start_sim_time and st.arrival_time<start_sim_time+TIME_SLOT]
            self.decision_maker.decide_allocation(current_requests, self.edge_nodes, self.models, self.completed_requests)
            start_sim_time += TIME_SLOT


    def plot_metrics(self, delays,  diff_time, diff_avg_acc):
        if delays:
            plt.figure(figsize=(10, 5))
            plt.hist(delays, bins=20, alpha=0.7, edgecolor='black')
            plt.title("Histogram of Request Response Times")
            plt.xlabel("Response Time (seconds)")
            plt.ylabel("Number of Requests")
            plt.grid(True)
            plt.show()
        else:
            print("No delays...")

            
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
        plt.title("QoS Deadline vs Actual Finish Time per Request Set")
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
            plt.title("QoS accuracy vs estimated avg accuracy for Request Sets with completed requests")
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

        
            

    def print_stats(self):
        total = NUM_REQUEST*len(self.request_sets)
        print("first request set:  ", self.request_sets[0])
        completed = sum([req.num_completed_request for req in self.request_sets])
        rejected = sum([NUM_REQUEST-req.num_completed_request for req in self.request_sets])
        delays =  [finish - arrival for arrival, _, finish in self.completed_requests]
        diff_time = [(req.qos_response_time - req.finish_time) for req in self.request_sets]
        diff_avg_acc = [(sum(req.estimated_accuracy)/req.num_completed_request)- req.required_accuracy for req in self.request_sets if req.num_completed_request>0]
        total_diff_time = sum([(req.qos_response_time - req.finish_time) for req in self.request_sets])
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

        for node in self.edge_nodes:
            used_flops = node.used_flops()
            flops_util = used_flops / (UTILIZATION_THRESHOLD * node.total_flops_capacity)
            print(f"Node {node.id} FLOPs Utilization: {flops_util:.2%}")

        self.plot_metrics(delays, diff_time, diff_avg_acc)


def main():

    models = [
        MLModel(name="yolov5n", required_flops=7.7, accuracy= 0.3),
        MLModel(name="yolov5s", required_flops=24, accuracy= 0.4),
        MLModel(name="yolov5m", required_flops=64.2, accuracy= 0.58),
        MLModel(name="yolov5l", required_flops=135.0, accuracy= 0.88),
        MLModel(name="yolov5x", required_flops=246.4, accuracy= 0.95)
    ]

    edge_num = 3
    print("edge_num:  ", edge_num)
    nodes = []
    for i in range(edge_num):
        nodes.append(EdgeNode(id=i, flops_capacity=FLOPS[i], flops_watt = FLOPS_WATT[i], energy_limit=random.randint(10**4, 10**6), models= models) ) #CHECK: energy_limit of edge_noedes assigning randomly??

    arrival_fn = lambda: random.uniform(1.0, 3.0) # requests' arrival rate

    decison_maker = DecisionMaker()

    sim = Simulator(edge_nodes=nodes, models=models, arrival_rate_fn=arrival_fn, duration=60, decision_maker = decison_maker)
    sim.run()
    sim.print_stats()

if __name__ == "__main__":
    main()
