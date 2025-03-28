"""
Note: I need to import the data from parameters.py
"""

import pyomo.environ as pyo
import importlib
import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt


import parameters
from parameters import *

from analysis import plot_solution_with_map, plot_solution_with_covering_radius, generate_facility_summary, display_selected_variables_CAC, plot_solution_with_model_dependent_covering_radius

# Change scenario dynamically
scenario_name = "nguenyyiel_terkidi_baseline"  # New scenario
parameters.scenario_name = scenario_name  # Update it in the session
#importlib.reload(parameters)  # Reload the module to apply changes


def model_mshlam_mar25_CAMPS_CAC(obj, optimum_values, eps, model_data):
    """
    Inputs:
        model_data is a dictionary that contains the following elements: 
            I (array): (indices of) demand points; eg, ['i1','i2',...,'i16']
            J (array): (indices of) candidate locations for HFs; eg, ['j1',...,'j9']
            J_HP (array): (indices of) candidate locations for HPs; eg, ['j2','j3','j4','j6','j7','j8']
            J_HC (array): (indices of) candidate locations for HCs; eg, ['j1','j3','j4','j5','j9']
            C (array): (indices of) refugee camps; eg, ['Nguenyyiel', 'Pinyudo', 'Jewi', 'Kule', 'Tierkidi', 'Pinyudo-II']
            I_c (dictionary of arrays): per each refugee camp c in C, the list of demand points i in I that are within refugee camp c; 
            J_c (dictionary of arrays): per each refugee camp c in C, the list of candidate locations for HPs j in J that are within refugee camp c; 
            S (array): (indices of) types of services; eg, ['basic','maternal1','maternal2']
            P (array): (indices of) types of health workers; eg, ['doctor','nurse','midwife']
            L (array): (indices of) levels of HFs; eg, ['hp','hc']
        
            n_HF (dict): per l in L, the number of HF of level l to locate

            Pi (dict): per i in I, the total population at zone i 
            r1 (dict): pero i in I and s in S, the daily demand rate from i for service s during HP's opening times
            r2 (dict): per i in I and s in S, the daily demand rate from i for service s outside of HP's opening times 
            d1 (dict): per i in I and s in S, the number of people from i daily demanding service s during HP's opening times (it is Pi[i]*r1[i,s])
            d2 (dict): per i in I and s in S, the number of people from i daily demanding service s outside of HP's opening times (it is Pi[i]*r2[i,s])
            
            t (DataFrame): travel times (distances) between pairs of location nodes, ie, between pairs in {I, J_HP, J_JC}   
            n_W (dict): per p in P, the number of health workers of type p to locate
            lb (dict): per p in P and l in L, the minimum number of health workers of type p that need to be present at an open HF of level l
            ub (dict): per p in P and l in L, the maximum number of health workers of type p that can be present at an open HF of level l
            a_HF (dict): per s in S and l in L, boolean operator indicating whether service s can be provided at a HF of level l (1) or not (0)
            a_W (dict): per p in P and s in S, boolean operator indicating whether health worker type p is able to deliver service s (1) or not (0)
            t1max (integer): maximum/coverage travel time (distance) from a demand point to the HF that is its first assignment
            t2max (integer): maximum travel time (distance) from a demand point to the HC that is its second assignment
            q (dict): per s in S, the service time for s
            h (dict): per p in P, the number of daily working hours for health worker of type p
        
        
    
    Returns: m
        
    """

    
    m = pyo.ConcreteModel('MSHLAM')
    
    # Function to remove all components from the model
    def remove_all_components(m):
        components = list(m.component_objects())
        for comp in components:
            m.del_component(comp)
    
    # Remove all components from the model
    remove_all_components(m)
    
    m.I = pyo.Set(initialize = model_data['I'])
    m.J = pyo.Set(initialize = model_data['J'])
    m.J_HP = pyo.Set(initialize = model_data['J_HP'])
    m.J_HC = pyo.Set(initialize = model_data['J_HC'])
    
    m.C = pyo.Set(initialize = model_data['C'])

    m.IC = pyo.Param(m.C, initialize = model_data['I_c'], within = pyo.Any)
    m.JC = pyo.Param(m.C, initialize = model_data['J_c'], within = pyo.Any)

    m.S = pyo.Set(initialize = model_data['S'])
    m.P = pyo.Set(initialize = model_data['P'])
    m.L = pyo.Set(initialize = model_data['L'])
    
    m.n_HF = pyo.Param(m.L, initialize = model_data['n_HF'], within = pyo.Integers)

    m.Pi = pyo.Param(m.I, initialize = model_data['Pi'], within = pyo.NonNegativeIntegers)
    m.r1 = pyo.Param(m.I, m.S, initialize = model_data['r1'], within = pyo.NonNegativeReals)
    m.r2 = pyo.Param(m.I, m.S, initialize = model_data['r2'], within = pyo.NonNegativeReals)
    m.d1 = pyo.Param(m.I, m.S, initialize = model_data['d1'], within = pyo.NonNegativeReals)
    m.d2 = pyo.Param(m.I, m.S, initialize = model_data['d2'], within = pyo.NonNegativeReals)
    
    m.t = pyo.Param(m.I, m.J, initialize={(i, j): model_data['t'].loc[i, j] for i in model_data['t'].index for j in model_data['t'].columns})
    
    m.n_W = pyo.Param(m.P, initialize = model_data['n_W'], within = pyo.Integers)
    m.lb = pyo.Param(m.P, m.L, initialize = model_data['lb'], within = pyo.Integers)
    m.ub = pyo.Param(m.P, m.L, initialize = model_data['ub'], within = pyo.Integers)
    
    m.a_HF = pyo.Param(m.S, m.L, initialize = model_data['a_HF'], within = pyo.Binary)
    m.a_W = pyo.Param(m.P, m.S, initialize = model_data['a_W'], within = pyo.Binary)
    
    m.t1_max = pyo.Param(initialize = model_data['t1max'], within = pyo.NonNegativeReals)
    m.t2_max = pyo.Param(initialize = model_data['t2max'], within = pyo.NonNegativeReals) 

    m.q = pyo.Param(m.S, initialize = model_data['q'], within = pyo.NonNegativeReals)
    m.h = pyo.Param(m.P, initialize = model_data['h'], within = pyo.NonNegativeReals)
    
    # Variables:
    m.y = pyo.Var(m.J, m.L, within = pyo.Binary)
    m.x1 = pyo.Var(m.I, m.J, within = pyo.Binary)
    m.x2 = pyo.Var(m.I, m.J, within = pyo.Binary)
    m.f1 = pyo.Var(m.I, m.J, m.S, within = pyo.NonNegativeIntegers) # Laura, to do: relax the integrality
    m.f2 = pyo.Var(m.I, m.J, m.S, within = pyo.NonNegativeIntegers) # Laura, to do: relax the integrality
    m.w = pyo.Var(m.J, m.P, within = pyo.NonNegativeIntegers)
    m.tau1max = pyo.Var(m.C, within = pyo.NonNegativeReals)
    m.tau2max = pyo.Var(m.C, within = pyo.NonNegativeReals)

    m.obj1 = pyo.Var(within = pyo.NonNegativeIntegers) # Laura, to do: relax the integrality
    m.obj2 = pyo.Var(within = pyo.NonNegativeReals)
    m.obj3 = pyo.Var(within = pyo.NonNegativeReals)
    


    # OBJECTIVE Function Equations
    @m.Constraint()
    def C_obj1(m):
        return m.obj1 == pyo.quicksum((m.f1[i, j, s] + m.f2[i,j,s]) for i in m.I for j in m.J for s in m.S)
    
    @m.Constraint()
    def C_obj2(m):
        return m.obj2 == pyo.quicksum(m.tau1max[c] for c in m.C)


    @m.Constraint()
    def C_obj3(m):
        return m.obj3 == pyo.quicksum(m.tau2max[c] for c in m.C)

    # Maximum values of the objectives to normalize the combined objective
    max_obj1 = sum(m.d1.values())  # (not tight) upper bound for obj1
    max_obj2 = max(m.t[i, j] for (i, j) in m.t)    # (not tight) upper bound for obj2
    max_obj3 = max(m.t[i, j] for i in m.I for j in m.J_HC)    # (not tight) upper bound for obj3

    # Now, define the combined objective function using these auxiliary variables.
    # Note that the original objectives had a maximization for satisfied demand and minimizations for max travel time and excess demand.
    
    if obj == "satisfied_demand":
        m.Objective = pyo.Objective(expr=m.obj1, sense=pyo.maximize)
    elif obj == "max_dist_first_assignment":
        m.Objective = pyo.Objective(expr=m.obj2, sense=pyo.minimize)
    elif obj == "max_dist_second_assignment":
        m.Objective = pyo.Objective(expr=m.obj3, sense=pyo.minimize)
    elif obj == "weighted_sum":
        m.Objective = pyo.Objective(
            expr=0.8 * (1 / max_obj1) * m.obj1 - 0.1 * (1 / max_obj2) * m.obj2 - 0.1 * (1 / max_obj3) * m.obj3, 
            sense=pyo.maximize
        )
    else:
        raise ValueError("Invalid objective selection")

    
    # Epsilon-CONSTRAINTS
    if obj != "max_dist_first_assignment" and "max_dist_first_assignment" in optimum_values:
        @m.Constraint()
        def O2_tau1max(m):
            return pyo.quicksum(m.tau1max[c] for c in m.C) <= (1+eps)*optimum_values["max_dist_first_assignment"]
    
    if obj != "max_dist_second_assignment" and "max_dist_second_assignment" in optimum_values: 
        @m.Constraint()
        def O3_tau2max(m):
            return pyo.quicksum(m.tau2max[c] for c in m.C) <= (1+eps)*optimum_values["max_dist_second_assignment"]
        
    if obj != "satisfied_demand" and "satisfied_demand" in optimum_values: 
        @m.Constraint()
        def O1_satisfieddemand(m):
            return pyo.quicksum((m.f1[i, j, s]+m.f2[i,j,s]) for i in m.I for j in m.J for s in m.S) >= (1-eps)* optimum_values["satisfied_demand"]
    
    
    # CONSTRAINTS
    @m.Constraint(m.L)
    def R1_budget_HFs(m, l):
        return pyo.quicksum(m.y[j,l] for j in m.J) <= m.n_HF[l]
    

    @m.Constraint(m.J, m.L)
    def R2_location_HFs(m, j, l):
        if (j not in m.J_HP and l == 'hp') or (j not in m.J_HC and l == 'hc'):
            return m.y[j, l] == 0
        return pyo.Constraint.Skip
    
    
    @m.Constraint(m.J)
    def R3_one_HF_per_location(m, j):
        return pyo.quicksum(m.y[j,l] for l in m.L) <= 1
        
  
    @m.Constraint(m.I)
    def R4_first_assignment(m, i):
        return pyo.quicksum(m.x1[i,j] for j in m.J) ==  1 

      
    @m.Constraint(m.C, m.I, m.J)
    def R5_first_assignment_only_within_camps(m, c, i, j):
        if ((i not in m.IC[c]) and (j in m.JC[c])) or ((i in m.IC[c]) and (j not in m.JC[c])):
            return m.x1[i,j] == 0 
        return pyo.Constraint.Skip


    @m.Constraint(m.C, m.I, m.J)
    def R6_maximum_distance_first_assignment(m, c, i, j):
        if (i in m.IC[c]) and (j in m.JC[c]):
            return m.t[i,j]*m.x1[i,j] <=  m.tau1max[c]
        return pyo.Constraint.Skip

    @m.Constraint(m.I, m.I, m.J)
    def R7_CAC_first_assignment(m, i1, i2, j):
        return (
            pyo.quicksum(m.x1[i1, k] for k in m.J if m.t[i1, k] > m.t[i1, j])
            + pyo.quicksum(m.x1[i2, k] for k in m.J if m.t[i1, k] <= m.t[i1, j] and m.t[i2, k] > m.t[i2, j])
            + pyo.quicksum(m.y[j, l] for l in m.L)
            <= 1
        )


    @m.Constraint(m.I)
    def R8_second_assignment(m, i):
        return pyo.quicksum(m.x2[i,j] for j in m.J) ==  1 
 

    @m.Constraint(m.C, m.I, m.J)
    def R9_second_assignment_only_within_camps(m, c, i, j):
        if ((i not in m.IC[c]) and (j in m.JC[c])) or ((i in m.IC[c]) and (j not in m.JC[c])):
            return m.x2[i,j] == 0 
        return pyo.Constraint.Skip
    

    @m.Constraint(m.I, m.I, m.J)
    def R10_CAC_second_assignment(m, i1, i2, j):
        return (
            pyo.quicksum(m.x2[i1, k] for k in m.J if m.t[i1, k] > m.t[i1, j])
            + pyo.quicksum(m.x2[i2, k] for k in m.J if m.t[i1, k] <= m.t[i1, j] and m.t[i2, k] > m.t[i2, j])
            + m.y[j, 'hc']
            <= 1
        )


    @m.Constraint(m.C, m.I, m.J)
    def R11_maximum_distance_second_assignment(m, c, i, j):
        if (i in m.IC[c]) and (j in m.JC[c]):
            return m.t[i,j] * m.x2[i,j] <= m.tau2max[c]
        return pyo.Constraint.Skip
    
    
    @m.Constraint(m.I, m.J) 
    def R12_first_assignment_is_HC(m, i, j):
        return 1-m.x2[i,j] <= (1-m.y[j,'hc']) + (1-m.x1[i,j])
    

    @m.Constraint(m.I, m.J)
    def R13_first_allocation_must_exist(m, i, j):
        return m.x1[i,j]  <= pyo.quicksum(m.y[j,l] for l in m.L)
    

    @m.Constraint(m.I, m.J)
    def R14_second_allocation_must_exist(m, i, j):
        return m.x2[i,j] <= m.y[j,'hc']
    
    
    @m.Constraint(m.I, m.J, m.S)
    def R15_relation_flow_first_assignment(m, i, j, s): 
        return m.f1[i,j,s] <= m.d1[i,s]*m.x1[i,j] 
    

    @m.Constraint(m.I, m.J, m.S)
    def R16_relation_flow_open_facility(m, i, j, s): 
        return m.f1[i,j,s] <= pyo.quicksum(m.d1[i,s] * m.a_HF[s,l] * m.y[j,l] for l in m.L) 


    @m.Constraint(m.I, m.J, m.S)
    def R17_relation_flow_second_assignment(m, i, j, s):
        return m.f2[i,j,s] <= m.d2[i,s]*m.x2[i,j]
    

    @m.Constraint(m.I, m.J, m.S)       
    def R18_relation_flow_open_HC(m, i, j, s): 
        return m.f2[i,j,s] <= pyo.quicksum(m.d2[i,s] * m.a_HF[s,l] * m.y[j,l] for l in m.L)
    

    @m.Constraint(m.J, m.S)
    def R19_satisfied_demand_HFs(m, j, s):
        return pyo.quicksum(m.f1[i,j,s] + m.f2[i,j,s] for i in m.I) <= (1/m.q[s]) * pyo.quicksum(m.h[p]*m.a_W[p,s]*m.w[j,p] for p in m.P)
    

    @m.Constraint(m.J)
    def R20_time_spent_demand_HFs(m, j):
        return pyo.quicksum(m.q[s] * (m.f1[i,j,s] + m.f2[i,j,s]) for i in m.I for s in m.S) <= pyo.quicksum(m.h[p]*m.w[j,p] for p in m.P)
      
 
    @m.Constraint(m.P)
    def R21_allocation_workers(m, p):
        return pyo.quicksum(m.w[j,p] for j in m.J) <= m.n_W[p]
    

    @m.Constraint(m.J, m.P)
    def R22_upper_bounds_workers(m, j, p):
        return m.w[j,p] <= pyo.quicksum(m.ub[p,l]*m.y[j,l] for l in m.L) 


    @m.Constraint(m.J, m.P)
    def R23_lower_bounds_workers(m, j, p):
        return pyo.quicksum(m.lb[p,l]*m.y[j,l] for l in m.L) <= m.w[j,p]
  

    return m



optimal_values = {}
eps = 0.2
solver = pyo.SolverFactory('cplex')
#solver.options['timelimit'] = 120
gaps = [0.05, 0.01, 0]
model = {} # store models for each gap
results = {} # store results for each gap

for gap in gaps:
    solver.options['mipgap'] = gap  

    # Function to compute optimal values for different objectives
    def get_optimal_values(optimal_values):
        for obj in ["max_dist_first_assignment","max_dist_second_assignment"]:
            model_ = model_mshlam_mar25_CAMPS_CAC(obj,optimal_values, eps, model_data)
            solver.solve(model_, tee=True)
            optimal_values[obj] = pyo.value(model_.Objective) # Store optimal value
        return optimal_values

    # Store optimal values for the current gap
    optimal_values[gap] = get_optimal_values({}) # Use a fresh dictionary for each gap

    # Create and solve the final model using the obtained optimal values
    model[gap] = model_mshlam_mar25_CAMPS_CAC("satisfied_demand", optimal_values[gap], eps, model_data)
    results[gap] = solver.solve(model[gap], tee=True)



# Plot the solution with a map background (only if our demand points and facilities have real-world coordinates; otherwise, use the code in the above cell)
plot_data = plot_solution_with_map(model[0], demand_points_gdf, hfs_gdf, show_arrows=False, show_HF_text=False)
plot_data.savefig("plot_data.pdf", format="pdf", bbox_inches="tight")


# Plot the open facilities
plot_sol_map = plot_solution_with_covering_radius(model[0], demand_points_gdf, hfs_gdf, cov_radius = 0) 
plot_sol_map.savefig("plot_sol_map.pdf", format="pdf", bbox_inches="tight")


# Example call with False for radii to prevent plotting
plot_sol_map_circles = plot_solution_with_model_dependent_covering_radius(model[0], demand_points_gdf, hfs_gdf, plot_first_cov_radius=True, plot_second_cov_radius=True, plot_color_first_assignment = True, plot_color_second_assignment = True)
plot_sol_map_circles.savefig("plot_sol_map_circles.pdf", format="pdf", bbox_inches="tight")


# Display only the selected variables

# Create a buffer to capture printed output
f = io.StringIO()
 
# Redirect stdout to the buffer and call your function
with redirect_stdout(f):
    display_selected_variables_CAC(model[0])

# Get the full string
output = f.getvalue()

# Save it to a text file
with open("display_selected_variables_CAC.txt", "w") as file:
    file.write(output)


# Create Summary table
generate_facility_summary(model[0])
# an html will be saved