from scipy.optimize import linear_sum_assignment
import numpy as np


def assign_requests_to_replicas_with_hungarian(request_sets, replicas):
    num_replicas = len(replicas)
    num_requests = len(request_sets)

    cost_matrix = np.full((num_replicas, num_requests), fill_value=np.inf)

    for i, rep in enumerate(replicas):
        for j, req in enumerate(request_sets):
            est_exec_time = rep.estimate_exec_time()
            start_time = max(req.arrival_time, rep.busy_until)
            finish_time = start_time + est_exec_time

            if finish_time <= req.qos_response_time and \
                    (sum(req.estimated_accuracy) + rep.model.accuracy) >= (
                    req.num_completed_request + 1) * req.required_accuracy:
                cost_matrix[i][j] = finish_time  # minimizes total finish time across all assignments
                #if want to maximize accuracy: cost_matrix[i][j] = -rep.model.accuracy
                #if include energy in cost
                #energy = rep.allocated_flops * rep.edge_node.flops_watt
                #cost_matrix[i][j] = 0.7 * finish_time + 0.3 * energy  # weighted sum


    # Hungarian requires finite values — if all entries are inf, we cannot assign
    if np.isinf(cost_matrix).all():
        return []

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r][c] != np.inf:
            assignments.append((r, c))  # (replica index, request index)

    return assignments


def decide_allocation(self, request_sets, edge_nodes, models, completed_requests):  # , rejected_requests):
    replicas_list = []
    total_model_flops = sum(m.required_flops for m in models)

    for node in edge_nodes:
        for model in models:
            model_weight = model.required_flops / total_model_flops
            flops = model_weight * node.total_flops_capacity * UTILIZATION_THRESHOLD  # CHECK: UTILIZATION_THRESHOLD ?
            energy = flops * node.flops_watt

            replicas_list.append({"edge_id": node.id,  # List of all potential containers CTN_j,l the system
                                  "model": model,
                                  "accuracy": model.accuracy,
                                  "service_time": model.required_flops / flops,
                                  "replica_flops": flops,
                                  "energy": energy,
                                  "current_replicas": len(node.replicas)
                                  })
    replicas_list.sort(key=lambda r: r["energy"])

    for ctn in replicas_list:
        matching_requests = [r for r in request_sets if
                             (NUM_REQUEST - r.num_completed_request) > 0 and sum(r.estimated_accuracy) + ctn[
                                 "accuracy"] >= (
                                         r.num_completed_request + 1) * r.required_accuracy]  # request sets with some uncompleted requests

        if not matching_requests:
            edge_nodes[ctn["edge_id"]].remove_idle_replicas(
                max(req.arrival_time for req in request_sets))  # just removing idle replicas from container
        else:
            edge_nodes[ctn["edge_id"]].remove_idle_replicas(
                max(req.arrival_time for req in matching_requests))  # remove_idle_replicas
            dominated_request = max(matching_requests, key=lambda r: (
                                                                                 NUM_REQUEST - r.num_completed_request) / r.qos_response_time)  # The set with maximum number of uncompleted requests and minimum deadline

            arrivals = [r.arrival_time for r in matching_requests]
            if len(arrivals) >= 2:
                inter_arrival_times = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
            elif len(arrivals) == 1:
                inter_arrival_times = [1.0] #avoid divide by zero by small positive value
            num_replica = self.decide_provisioning(dominated_request, inter_arrival_times, ctn,
                                                   edge_nodes)  # Provisioning decision

            if num_replica > 0:
                for _ in range(num_replica):
                    edge_nodes[ctn["edge_id"]].add_replica(ctn["model"])

            # Collect all available replicas across nodes
            available_replicas = []
            for node in edge_nodes:
                available_replicas.extend(rep for rep in node.replicas if
                                          rep.busy_until <= max(req.arrival_time for req in matching_requests))

            # Use Hungarian algorithm to assign available replicas to request sets
            assignments = assign_requests_to_replicas_with_hungarian(matching_requests, available_replicas)

            # Process requests based on optimal assignments
            for rep_idx, req_idx in assignments:
                replica = available_replicas[rep_idx]
                req_set = matching_requests[req_idx]

                start, finish, accuracy = replica.process_request(req_set.arrival_time)

                if finish <= req_set.qos_response_time and \
                        accuracy + sum(req_set.estimated_accuracy) >= req_set.required_accuracy * (
                        req_set.num_completed_request + 1):
                    completed_requests.append((req_set.arrival_time, start, finish))
                    req_set.estimated_accuracy.append(accuracy)
                    req_set.finish_time = max(req_set.finish_time, finish)
                    req_set.num_completed_request += 1

