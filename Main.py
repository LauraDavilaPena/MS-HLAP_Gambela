import Location_Problem as problem
import pygmo as pg
import time

# Laura: I added this:
import importlib
import parameters
from parameters import *
# Change scenario dynamically
scenario_name = "pugnido_baseline"  
parameters.scenario_name = scenario_name  
importlib.reload(parameters)  
##################################################################

# Record the start time
start_time = time.time()

pop_size =20
gen = 10
model_data = parameters.model_data # Laura: I added this 
p = pg.problem(problem.Location_Problem(model_data))
pop = pg.population(prob=p, size=pop_size, seed=12314876)
algo = pg.algorithm(pg.nsga2(gen=gen))
pop = algo.evolve(pop)


end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
# print(f"Elapsed Time: {elapsed_time} seconds")

# Save the best solution
Best_soluton = pop.get_x()[0]
original_p = p.extract(problem.Location_Problem)
KPIs = original_p.fitness(Best_soluton)
open_hps, open_hcs = original_p.decode_facilities(Best_soluton)