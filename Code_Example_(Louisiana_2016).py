import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain, proposals, constraints, accept, Election)
from gerrychain.updaters import Tally
from gerrychain.proposals import recom
from gerrychain.metrics import polsby_popper
from functools import partial
import pandas
import statistics
import numpy as np

def weighted_wasted_votes(weight, party1_votes, party2_votes):
    total_votes = party1_votes + party2_votes
    if party1_votes > party2_votes:
        party1_waste = weight * (party1_votes - total_votes / 2)
        party2_waste = party2_votes
    else:
        party1_waste = party1_votes
        party2_waste = weight * (party2_votes - total_votes / 2)
    return party1_waste, party2_waste

def weighted_wasted_votes_one(weight, party1_votes, party2_votes):
    total_votes = party1_votes + party2_votes
    if party1_votes > party2_votes:
        party1_waste = weight * (party1_votes - total_votes / 2)
    else:
        party1_waste = party1_votes
    return party1_waste

def weighted_wasted_votes_two(weight, party1_votes, party2_votes):
    total_votes = party1_votes + party2_votes
    if party1_votes > party2_votes:
        party2_waste = party2_votes
    else:
        party2_waste = weight * (party2_votes - total_votes / 2)
    return party2_waste

def weighted_efficiency_gap(results, weight):
    party1, party2 = [results.counts(party) for party in results.election.parties]
    weights = [weight for i in party1]
    wasted_votes_by_part = map(weighted_wasted_votes, weights, party1, party2)
    total_votes = results.total_votes()
    num = sum(waste2-waste1 for waste1, waste2 in wasted_votes_by_part)
    return num / total_votes

def relative_efficiency_gap(results, weight):
    party1, party2 = [results.counts(party) for party in results.election.parties]
    weights = [weight for i in party1]
    wasted_votes_by_part_one = map(weighted_wasted_votes_one, weights, party1, party2)
    wasted_votes_by_part_two = map(weighted_wasted_votes_two, weights, party1, party2)
    num1 = sum(i for i in wasted_votes_by_part_one)
    dem1 = sum(i for i in party1)
    num2 = sum(i for i in wasted_votes_by_part_two)
    dem2 = sum(i for i in party2)
    return num2 / dem2 - num1 / dem1

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def declination(results):
    party1, party2 = [results.counts(party) for party in results.election.parties]
    shares1 = []
    for i in range(0, len(party1)):
        shares1.append(party1[i] / (party1[i] + party2[i]))
    shares1.sort()
    dems = []
    reps = []
    for i in shares1:
        if i <= 0.5:
            reps.append(i)
        else:
            dems.append(i)
    if len(reps) == 0 or len(dems) == 0:
        return 100
    theta = np.arctan((1 - 2 * np.mean(reps)) * len(party1) / len(reps))
    gamma = np.arctan((2 * np.mean(dems) - 1) * len(party1) / len(dems))
    return 2.0 * (gamma - theta) / 3.1415926535

graph = Graph.from_file("./LA_1519.shp", reproject=False)
election = Election("PRES16", {"Dem": "PRES16D", "Rep": "PRES16R"})
initial_partition = GeographicPartition(
   graph,
   assignment = "CD",
   updaters = {
       "population": Tally("TOTPOP", alias="population"),
       "PRES16": election
   }
)
ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
proposal = partial(recom,
                  pop_col = "TOTPOP",
                  pop_target = ideal_population,
                  epsilon = 0.02,
                  node_repeats = 2)

compactness_bound = constraints.UpperBound(
   lambda p: len(p["cut_edges"]),
   2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)
chain = MarkovChain(
   proposal = proposal,
   constraints=[
       pop_constraint,
       compactness_bound
   ],
   accept = accept.always_accept,
   initial_state = initial_partition,
   total_steps = 3000,
)

data = pandas.DataFrame(
   sorted(partition["PRES16"].percents("Dem")) for partition in chain
)

print("Efficiency Gaps:\n")
for partition in chain:
    print(partition["PRES16"].efficiency_gap())
print("Mean Medians:\n")
for partition in chain:
    print(partition["PRES16"].mean_median())
print("Partisan Bias:\n")
for partition in chain:
    print(partition["PRES16"].partisan_bias())
print("Weighted Efficiency Gap (2) \n")
for partition in chain:
    print(weighted_efficiency_gap(partition["PRES16"],2))
print("Relative Efficiency Gap (1) \n")
for partition in chain:
    print(relative_efficiency_gap(partition["PRES16"],1))
print("Relative Efficiency Gap (2) \n")
for partition in chain:
    print(relative_efficiency_gap(partition["PRES16"],2))
print("Declination \n")
for partition in chain:
    print(declination(partition["PRES16"]))

fig, ax = plt.subplots(figsize=(8,6))
ax.axhline(0.5, color="#cccccc")
data.boxplot(ax=ax, positions=range(len(data.columns)))
plt.plot(data.iloc[0], "ro")

ax.set_title("Comparing 2016 NM Presidential Election to Ensemble")
ax.set_ylabel("Democratic vote % (PRES 2016)")
ax.set_xlabel("Sorted districts")
ax.set_ylim(0.4, 0.7)
ax.set_yticks([0.4,0.45,0.5,0.5,0.6,0.65,0.7])
plt.show()

