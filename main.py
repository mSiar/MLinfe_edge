import random
import numpy as np
import matplotlib.pyplot as plt
from setting import ENERGY_PER_FLOP, UTILIZATION_THRESHOLD, MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY, MIN_RESPONSE_TIME, MAX_RESPONSE_TIME


class MLModel:
    def __init__(self, name, required_flops, accuracy):
        self.name = name
        self.required_flops = required_flops
        self.accuracy = accuracy #this not dependent on allocated FLOPs, hence static

    def estimate_exec_time(self, allocated_flops):
        return self.required_flops / allocated_flops


class Replica:
    # Change it according to the relation of flops and energy consumption per flop equation

    def __init__(self, allocated_flops, model: MLModel):
        self.allocated_flops = allocated_flops
        self.model = model
        self.busy_until = 0

    def process_request(self, current_time):
        exec_time = self.model.estimate_exec_time(self.allocated_flops)
        start_time = max(current_time, self.busy_until)
        finish_time = start_time + exec_time
        self.busy_until = finish_time
        return start_time, finish_time

    def energy_consumption(self):
        return self.allocated_flops * ENERGY_PER_FLOP


class EdgeNode:
    def __init__(self, id, flops_capacity, energy_limit):
        self.id = id
        self.total_flops_capacity = flops_capacity
        self.total_energy_limit = energy_limit
        self.available_flops = UTILIZATION_THRESHOLD * flops_capacity
        self.available_energy = energy_limit
        self.replicas = []

    def can_allocate(self, flops_needed):

        #total_allocated = sum(rep.allocated_flops for rep in self.replicas)
        #return (total_allocated + flops_needed) <= 0.8 * self.flops_capacity
        return flops_needed <= self.available_flops

    def current_energy_usage(self):
        return sum(rep.energy_consumption() for rep in self.replicas)

    def under_energy_limit(self):
        return self.current_energy_usage() <= self.energy_limit

    def used_flops(self):
        return UTILIZATION_THRESHOLD* self.total_flops_capacity - self.available_flops

    def used_energy(self):
        return self.total_energy_limit - self.available_energy

    def add_replica(self, flops, model):
        estiamated_energy = flops* ENERGY_PER_FLOP
        if self.can_allocate(flops) and self.available_energy >= estiamated_energy:
            replica = Replica(flops, model)
            self.replicas.append(replica)
            self.available_flops -= flops
            self.available_energy -= estiamated_energy
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

class Request:
    def __init__(self, arrival_time, qos_accuracy, qos_response_time):
        self.arrival_time = arrival_time
        self.required_accuracy = qos_accuracy
        self.qos_response_time = qos_response_time


class DecisionMaker:
    def __init__(self, default_flops_per_replica=10):
        self.default_flops_per_replica = default_flops_per_replica

    def decide_allocation(self, edge_nodes, model, inter_arrival_times, qos_target):
        lambda_rate = 1 / np.mean(inter_arrival_times)
        best_option = None
        print(f"[Debug] Inter arrival times= {inter_arrival_times}")

        for flops in range(10, 101, 10):  # Try from 10 to 100 FLOPs per replica
            service_time_sample = [model.estimate_exec_time(flops) for _ in range(100)]
            print(f"[Debug] Service time={service_time_sample}")
            mu = 1 / np.mean(service_time_sample)
            ca2 = np.var(inter_arrival_times) / (np.mean(inter_arrival_times) ** 2)
            cs2 = np.var(service_time_sample) / (np.mean(service_time_sample) ** 2)

            for c in range(1, 100):
                rho = lambda_rate / (c * mu)
                if rho >= 1:
                    continue
                wq = ((ca2 + cs2) / 2) * (rho ** (np.sqrt(2 * (c + 1)) - 1)) / (c * (1 - rho)) / mu
                tr = wq + 1 / mu
                print(f"[Debug] Trying FLOPs={flops}, c={c}, estimated response time={tr:.2f}, target={qos_target:.2f}")
                if tr <= qos_target:
                    best_option = (c, flops)
                    break  # Accept the first feasible option per FLOPs level

            if best_option:
                break

        return best_option



class Simulator:
    def __init__(self, edge_nodes, models, arrival_rate_fn, duration, decision_maker):
        self.edge_nodes = edge_nodes
        self.models = models
        self.arrival_rate_fn = arrival_rate_fn
        self.duration = duration

        self.requests = []
        self.completed_requests = []
        self.decision_maker = decision_maker
        self.rejected_requests =0

    def generate_requests(self):
        current_time = 0
        while current_time < self.duration:
            inter_arrival = self.arrival_rate_fn()
            current_time += inter_arrival
            required_accuracy = round(random.uniform(MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY),2)
            qos_response_time = round(random.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME),2)
            self.requests.append(Request(current_time, required_accuracy, qos_response_time))


    def allocate_replicas(self):
        print(" [Debug] Allocating  replicas...")
        #Group request by model that can serve them (based accuracy)
        model_to_requests = {
            model: [r for r in self.requests if r.required_accuracy <= model.accuracy]
            for model in self.models
        }

        for model, matching_requests in model_to_requests.items():
            allocated_count =0
            print(f"[Debug] Evaluating model: {model.name}, matching requests: {len(matching_requests)}")
            arrivals = [r.arrival_time for r in matching_requests]
            if len(arrivals) < 2:
                continue
            inter_arrival_times = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
            avg_qos_target = np.mean([r.qos_response_time for r in matching_requests])
            print(f"[Debug] Avg QoS target for model {model.name}: {avg_qos_target:.2f}")
            decision = self.decision_maker.decide_allocation(self.edge_nodes, model, inter_arrival_times,
                                                             avg_qos_target)

            if decision is None:
                print(f"Warning: Unable to satisfy QoS for model {model.name} with available resources.")
                continue
            print(f"[Debug] Total replicas allocated for model {model.name}: {allocated_count}")
            num_replicas, flops_per_replica = decision
            replicas_per_node = max(1, num_replicas // len(self.edge_nodes))

            for node in self.edge_nodes:
                for _ in range(replicas_per_node):
                    replica = node.add_replica(flops_per_replica, model)
                    if replica:
                        print(f"[Debug] Replica added on Node {node.id} for model {model.name}")
                        allocated_count += 1
                    else:
                        print(f"[Warning] Could not allocate replica on Node {node.id} for model {model.name}")




    def route_request(self, request):
        best_replica = None
        best_start_time = float('inf')
        for node in self.edge_nodes:
            node.remove_idle_replicas(request.arrival_time)
            for replica in node.replicas:
                start_time, _ = replica.process_request(request.arrival_time)
                if start_time < best_start_time:
                    best_start_time = start_time
                    best_replica = replica

        if best_replica:
            start, finish = best_replica.process_request(request.arrival_time)
            self.completed_requests.append((request.arrival_time, start, finish))
        else:
            self.rejected_requests += 1

    def run(self):
        self.generate_requests()
        self.allocate_replicas()
        for request in self.requests:
            self.route_request(request)

    def plot_metrics(self, delays):
        if not delays:
            return
        plt.figure(figsize=(10, 5))
        plt.hist(delays, bins=20, alpha=0.7, edgecolor='black')
        plt.title("Histogram of Request Response Times")
        plt.xlabel("Response Time (seconds)")
        plt.ylabel("Number of Requests")
        plt.grid(True)
        plt.show()

    def print_stats(self):
        total = len(self.requests)
        completed = len(self.completed_requests)
        delays =  [finish - arrival for arrival, _, finish, _ in self.completed_requests]


        print(f"Total Requests: {total}")
        print(f"Completed Requests: {completed}")
        print(f"Rejected Requests: {self.rejected_requests} ({self.rejected_requests / total:.2%})")

        if completed > 0:
            avg_delay = delays / total
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

        self.plot_metrics(delays)






if __name__ == "__main__":
    nodes = [
        EdgeNode(id=1, flops_capacity=10000, energy_limit=8000),
        EdgeNode(id=2, flops_capacity=8000, energy_limit=6000),
    ]

    models = [
        MLModel(name="ML1", required_flops=10, accuracy= 0.6),
        MLModel(name="ML2", required_flops=20, accuracy= 0.7),
        MLModel(name="ML3", required_flops=30, accuracy= 0.9)
    ]

    arrival_fn = lambda: random.uniform(1.0, 3.0)

    decison_maker = DecisionMaker(default_flops_per_replica=100)

    sim = Simulator(edge_nodes=nodes, models=models, arrival_rate_fn=arrival_fn, duration=60, decision_maker = decison_maker)
    sim.run()
    sim.print_stats()
