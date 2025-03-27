import numpy as np
import pyomo.environ as pyo

class Location_Problem():
    def __init__(self, model_data):
        self.model_data = model_data
        self.n = len(model_data['J'])
        self.J_HP = set(model_data['J_HP']) 
        self.J_HC = set(model_data['J_HC'])  
        self.J = model_data['J']
        self.J_c = model_data['J_c'] 
        self.max_hps = model_data['n_HF']['hp'] 
        self.max_hcs = model_data['n_HF']['hc'] 

    def get_bounds(self):
        return (np.full(self.n,0), np.full(self.n,3))
    
    def get_nobj(self):
        return 3
    
    def decode_facilities(self, X):
        open_hps = []
        open_hcs = []
        chosen_hcs = set()

        # 1. Ensure at least one HC per camp (highest X value among eligible HCs)
        for camp, camp_locs in self.J_c.items():  
            hc_locs = [j for j in camp_locs if j in self.J_HC] 
            if hc_locs:
                best_j = max(hc_locs, key=lambda j: X[np.where(self.J==j)[0][0]]) 
                                                                         
                open_hcs.append(best_j)
                chosen_hcs.add(best_j)

        # 2. Select other HCs (if under max limit)
        hc_candidates = sorted(
            [j for j in self.J_HC if j not in chosen_hcs],
            key=lambda j: X[np.where(self.J==j)[0][0]],  
            reverse=True
        )
        for j in hc_candidates:
            if len(open_hcs) >= self.max_hcs: 
                                                         
                break
            val = X[np.where(self.J==j)[0][0]]
            if 2 <= val <= 3:
                open_hcs.append(j)

        # 3. Select HPs based on value in [1,2) 
        hp_candidates = sorted(
            [j for j in self.J_HP], 
            key=lambda j: X[np.where(self.J==j)[0][0]], 
            reverse=True
        )
        for j in hp_candidates:
            if len(open_hps) >= self.max_hps: 
                break
            val = X[np.where(self.J==j)[0][0]]
            if 1 <= val < 2: 
                open_hps.append(j)

        return open_hps, open_hcs

    
    def fitness(self, X):
        
        """
        Compute the fitness of a given facility configuration by assigning demand and solving an LP.
        
        Inputs:
            Encoded solution X (list or np.array): Vector of length |J| with values in [0, 3]
        Returns:
            satisfied_demand (float): Total satisfied demand
            lost_demand (float): Total lost demand
            max_dist_first_assignment (float): Maximum distance in first assignment
            max_dist_second_assignment (float): Maximum distance in second assignment
        """

        # Step 1: Assign demand points to nearest open facilities
        model_data = self.model_data
        open_hps, open_hcs = self.decode_facilities(X)
        
        first_assignment = {}  # i -> j (first assignment)
        second_assignment = {}  # i -> j (second assignment)
        max_dist_first_per_camp = {}
        max_dist_second_per_camp = {}

        open_facilities = np.concatenate([open_hps, open_hcs])

        for c in model_data['C']: # Iterate over each refugee camp
            I_c = model_data['I_c'][c] # Demand points in camp c
            J_c = np.array([j for j in open_facilities if j in model_data['J_c'][c]], dtype=object)  # Open facilities in camp c as an array

            if J_c.size == 0:  # If no facilities are open in this camp, skip assignment
                max_dist_first_per_camp[c] = 0
                max_dist_second_per_camp[c] = 0
                continue

            max_dist_first_per_camp[c] = 0
            max_dist_second_per_camp[c] = 0


            for i in I_c:  

                # First assignment: Closest HP or HC in the same camp
                open_hfs_c = np.array([j for j in open_facilities if j in J_c], dtype=object)  # Open HPs in camp c as an array

                if open_hfs_c.size > 0:  # If there are open HCs in this camp
                    closest_first = open_hfs_c[np.argmin([model_data['t'].loc[i, j] for j in open_hfs_c])]
                    first_assignment[i] = closest_first
                    max_dist_first_per_camp[c] = max(max_dist_first_per_camp[c], model_data['t'].loc[i, closest_first])
                else:
                    first_assignment[i] = None  # No valid second assignment possible

                # Second assignment: Closest HC in the same camp
                open_hcs_c = np.array([j for j in open_hcs if j in J_c], dtype=object)  # Open HCs in camp c as an array

                if open_hcs_c.size > 0:  # If there are open HCs in this camp
                    closest_second = open_hcs_c[np.argmin([model_data['t'].loc[i, j] for j in open_hcs_c])]
                    second_assignment[i] = closest_second
                    max_dist_second_per_camp[c] = max(max_dist_second_per_camp[c], model_data['t'].loc[i, closest_second])
                else:
                    second_assignment[i] = None  # No valid second assignment possible

        # Compute the sum of the maximum distances across all camps
        sum_max_dist_first_assignment = sum(max_dist_first_per_camp.values())
        sum_max_dist_second_assignment = sum(max_dist_second_per_camp.values())

        # Step 2: Solve LP for Workforce Allocation

        m = self.create_workforce_model(open_hps, open_hcs, open_facilities, first_assignment, second_assignment)

        # Solve LP
        solver = pyo.SolverFactory('cplex')
        solver.solve(m)

        # Compute results
        satisfied_demand = pyo.value(m.obj)

        return [-satisfied_demand, sum_max_dist_first_assignment, sum_max_dist_second_assignment]
    
    def create_workforce_model(self, open_hps, open_hcs, open_facilities, first_assignment, second_assignment):
        model_data = self.model_data
        m = pyo.ConcreteModel()

        # Function to remove all components from the model
        def remove_all_components(m):
            components = list(m.component_objects())
            for comp in components:
                m.del_component(comp)
        
        # Remove all components from the model
        remove_all_components(m)

        # Define sets
        m.I = pyo.Set(initialize = model_data['I'])  # demand points
        m.J = pyo.Set(initialize = open_facilities)  # Only open facilities
        m.P = pyo.Set(initialize = model_data['P'])  # types of professionals
        m.S = pyo.Set(initialize = model_data['S'])  # services
        m.L = pyo.Set(initialize = model_data['L'])  # types of facilities

        # Define parameters
        m.n_HF = pyo.Param(m.L, initialize = model_data['n_HF'], within = pyo.Integers)
        m.d1 = pyo.Param(m.I, m.S, initialize = model_data['d1'], within = pyo.NonNegativeReals)
        m.d2 = pyo.Param(m.I, m.S, initialize = model_data['d2'], within = pyo.NonNegativeReals)
        m.t = pyo.Param(m.I, m.J, initialize = {(i, j): model_data['t'].loc[i, j] for i in model_data['t'].index for j in m.J})
        m.n_W = pyo.Param(m.P, initialize = model_data['n_W'], within = pyo.Integers)
        m.lb = pyo.Param(m.P, m.L, initialize = model_data['lb'], within = pyo.Integers)
        m.ub = pyo.Param(m.P, m.L, initialize = model_data['ub'], within = pyo.Integers)
        m.a_HF = pyo.Param(m.S, m.L, initialize = model_data['a_HF'], within = pyo.Binary)
        m.a_W = pyo.Param(m.P, m.S, initialize = model_data['a_W'], within = pyo.Binary)
        m.q = pyo.Param(m.S, initialize = model_data['q'], within = pyo.NonNegativeReals)
        m.h = pyo.Param(m.P, initialize = model_data['h'], within = pyo.NonNegativeReals)

        # Define decision variables
        m.w = pyo.Var(m.J, m.P, within = pyo.NonNegativeIntegers)
        m.f1 = pyo.Var(m.I, m.J, m.S, within = pyo.NonNegativeIntegers)
        m.f2 = pyo.Var(m.I, m.J, m.S, within = pyo.NonNegativeIntegers)

        # Define the objective
        m.obj = pyo.Objective(
            expr=pyo.quicksum(m.f1[i, j, s] + m.f2[i, j, s] for i in m.I for j in m.J for s in m.S),
            sense=pyo.maximize
        )

        # Constraint definitions (e.g., demand satisfaction, worker capacity, etc.)

        
        @m.Constraint(m.I, m.J, m.S)
        def R15_relation_flow_first_assignment(m, i, j, s):
            return m.f1[i, j, s] <= m.d1[i, s] * (1 if first_assignment[i] == j else 0)
        

        @m.Constraint(m.I, m.J, m.S)
        def R16_relation_flow_open_facility(m, i, j, s):
            if j in open_hps:
                return m.f1[i, j, s] <= m.d1[i, s] * m.a_HF[s, 'hp']
            elif j in open_hcs:
                return m.f1[i, j, s] <= m.d1[i, s] * m.a_HF[s, 'hc']
            return pyo.Constraint.Skip
        

        @m.Constraint(m.I, m.J, m.S)
        def R17_relation_flow_second_assignment(m, i, j, s):
            return m.f2[i, j, s] <= m.d2[i, s] * (1 if second_assignment[i] == j else 0)


        @m.Constraint(m.I, m.J, m.S)
        def R18_relation_flow_open_HC(m, i, j, s):
            if j in open_hcs:
                return m.f2[i, j, s] <= m.d2[i, s] * m.a_HF[s, 'hc']
            return pyo.Constraint.Skip


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
            if j in open_hps:
                return m.w[j, p] <= m.ub[p, 'hp']
            elif j in open_hcs:
                return m.w[j, p] <= m.ub[p, 'hc']
            return pyo.Constraint.Skip


        @m.Constraint(m.J, m.P)
        def R23_lower_bounds_workers(m, j, p):
            if j in open_hps:
                return m.lb[p,'hp'] <= m.w[j,p]
            elif j in open_hcs:
                return m.lb[p,'hc'] <= m.w[j, p] 
            return pyo.Constraint.Skip
        

        return m
