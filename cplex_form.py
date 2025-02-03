import geopandas as gpd
import numpy as np
import pandas as pd
import itertools
import cplex
import matplotlib.pyplot as plt
from parameters import (
    location_nodes,
    demand_points_gdf,
    hfs_gdf,
    hps_gdf,
    hcs_gdf,
    dps,
    hfs,
    hps,
    hcs,
    services,
    health_workers,
    levels,
    HFs_to_locate,
    dd_oh,
    dd_ch,
    distance_matrix,
    workers_to_allocate,
    lb_workers,
    a_HF,
    a_W,
    t1max,
    service_time,
    working_hours,
    get_nearby_HFs
)

# =============================================================================
# SETS AND PARAMETERS
# =============================================================================

# Sets
I = dps            # demand points
J = hfs            # candidate locations for health facilities (HFs)
J_HP = hps         # candidate locations for HPs
J_HC = hcs         # candidate locations for HCs
S = services       # types of services
P = health_workers # types of health workers
L = levels         # levels of HFs

# Parameters
n_HF = dict(zip(levels, HFs_to_locate))
n_W = dict(zip(health_workers, workers_to_allocate))
d1 = dd_oh
d2 = dd_ch
lb = lb_workers
q = dict(zip(services, service_time))
c_work = dict(zip(health_workers, working_hours))
S_l = {l: {s for s in S if a_HF.get((s, l), 0) == 1} for l in L}
t = distance_matrix  # travel times between demand points and HFs
J_i = get_nearby_HFs(distance_matrix, dps, t1max)  # for each i, list of j within t1max

# =============================================================================
# BUILDING THE CPLEX MODEL
# =============================================================================

# We will build our model by first creating all decision variables.
# To mimic the Pyomo model, we define:
#   y[j,l]       binary (HF at location j at level l)
#   x1[i,j,s]    binary (first assignment)
#   x2[i,j,s]    binary (second assignment; only for j in J_HC)
#   f1[i,j,s]    integer (flow from first assignment)
#   f2[i,j,s]    integer (flow from second assignment)
#   z[j,s]       binary (services provided at facility j for service s)
#   w[j,p,s]     integer (number of workers of type p assigned to j for service s)
#   t2max        continuous (maximum travel time for second assignment)
#   delta[j,s]   integer (unmet demand at facility j for service s)
#   deltamax     integer (maximum over j of total unmet demand)

var_names = []
var_obj = []    # objective coefficients
var_types = []  # "B" for binary, "I" for integer, "C" for continuous
var_lb = []     # lower bounds
var_ub = []     # upper bounds

# Dictionaries to keep track of the index of each variable
y_vars = {}     # key: (j, l)
x1_vars = {}    # key: (i, j, s)
x2_vars = {}    # key: (i, j, s) with j in J_HC
f1_vars = {}    # key: (i, j, s)
f2_vars = {}    # key: (i, j, s) for j in J_HC
z_vars = {}     # key: (j, s)
w_vars = {}     # key: (j, p, s)
delta_vars = {} # key: (j, s)

# --------------------------
# Variables y[j,l]: binary; objective coeff = +0.5
# R2: if (j not in J_HP and l == 'hp') or (j not in J_HC and l == 'hc'), then y[j,l] = 0.
# --------------------------
for j in J:
    for l in L:
        name = f"y_{j}_{l}"
        obj_val = 0.5
        vartype = "B"
        lb_val = 0.0
        ub_val = 1.0
        # Enforce constraint R2 by fixing to zero if the location j is not eligible for level l.
        if (l == 'hp' and j not in J_HP) or (l == 'hc' and j not in J_HC):
            ub_val = 0.0
        y_vars[(j, l)] = len(var_names)
        var_names.append(name)
        var_obj.append(obj_val)
        var_types.append(vartype)
        var_lb.append(lb_val)
        var_ub.append(ub_val)

# --------------------------
# Variables x1[i,j,s]: binary; objective coeff = 0
# R5: if j is not in the list of reachable facilities for i, then x1[i,j,s] = 0.
# --------------------------
for i in I:
    for j in J:
        for s in S:
            name = f"x1_{i}_{j}_{s}"
            obj_val = 0.0
            vartype = "B"
            lb_val = 0.0
            ub_val = 1.0
            if j not in J_i.get(i, []):
                ub_val = 0.0  # fix variable to zero if j is unreachable from i
            x1_vars[(i, j, s)] = len(var_names)
            var_names.append(name)
            var_obj.append(obj_val)
            var_types.append(vartype)
            var_lb.append(lb_val)
            var_ub.append(ub_val)

# --------------------------
# Variables x2[i,j,s]: binary; objective coeff = 0
# --------------------------
for i in I:
    for j in J_HC:  # x2 only for j in J_HC
        for s in S:
            name = f"x2_{i}_{j}_{s}"
            obj_val = 0.0
            vartype = "B"
            lb_val = 0.0
            ub_val = 1.0
            x2_vars[(i, j, s)] = len(var_names)
            var_names.append(name)
            var_obj.append(obj_val)
            var_types.append(vartype)
            var_lb.append(lb_val)
            var_ub.append(ub_val)

# --------------------------
# Variables f1[i,j,s]: integer; objective coeff = 0
# f1[i,j,s] = d1[i,s]*x1[i,j,s]  (R15)
# --------------------------
for i in I:
    for j in J:
        for s in S:
            name = f"f1_{i}_{j}_{s}"
            obj_val = 0.0
            vartype = "I"
            lb_val = 0.0
            ub_val = cplex.infinity
            f1_vars[(i, j, s)] = len(var_names)
            var_names.append(name)
            var_obj.append(obj_val)
            var_types.append(vartype)
            var_lb.append(lb_val)
            var_ub.append(ub_val)

# --------------------------
# Variables f2[i,j,s]: integer; objective coeff = 0
# f2[i,j,s] = d2[i,s]*x2[i,j,s]  (R16)
# --------------------------
for i in I:
    for j in J_HC:
        for s in S:
            name = f"f2_{i}_{j}_{s}"
            obj_val = 0.0
            vartype = "I"
            lb_val = 0.0
            ub_val = cplex.infinity
            f2_vars[(i, j, s)] = len(var_names)
            var_names.append(name)
            var_obj.append(obj_val)
            var_types.append(vartype)
            var_lb.append(lb_val)
            var_ub.append(ub_val)

# --------------------------
# Variables z[j,s]: binary; objective coeff = 0
# --------------------------
for j in J:
    for s in S:
        name = f"z_{j}_{s}"
        obj_val = 0.0
        vartype = "B"
        lb_val = 0.0
        ub_val = 1.0
        z_vars[(j, s)] = len(var_names)
        var_names.append(name)
        var_obj.append(obj_val)
        var_types.append(vartype)
        var_lb.append(lb_val)
        var_ub.append(ub_val)

# --------------------------
# Variables w[j,p,s]: integer; objective coeff = 0
# --------------------------
for j in J:
    for p in P:
        for s in S:
            name = f"w_{j}_{p}_{s}"
            obj_val = 0.0
            vartype = "I"
            lb_val = 0.0
            ub_val = cplex.infinity
            w_vars[(j, p, s)] = len(var_names)
            var_names.append(name)
            var_obj.append(obj_val)
            var_types.append(vartype)
            var_lb.append(lb_val)
            var_ub.append(ub_val)

# --------------------------
# Variable t2max: continuous; objective coeff = 0
# (R10: t2max >= t[i,j]*x2[i,j,s] for all i, j, s)
# --------------------------
t2max_index = len(var_names)
var_names.append("t2max")
var_obj.append(0.0)
var_types.append("C")
var_lb.append(0.0)
var_ub.append(cplex.infinity)

# --------------------------
# Variables delta[j,s]: integer; objective coeff = 0
# --------------------------
for j in J:
    for s in S:
        name = f"delta_{j}_{s}"
        obj_val = 0.0
        vartype = "I"
        lb_val = 0.0
        ub_val = cplex.infinity
        delta_vars[(j, s)] = len(var_names)
        var_names.append(name)
        var_obj.append(obj_val)
        var_types.append(vartype)
        var_lb.append(lb_val)
        var_ub.append(ub_val)

# --------------------------
# Variable deltamax: integer; objective coeff = -0.5
# (R19: deltamax >= sum_s delta[j,s] for all j)
# --------------------------
deltamax_index = len(var_names)
var_names.append("deltamax")
var_obj.append(-0.5)
var_types.append("I")
var_lb.append(0.0)
var_ub.append(cplex.infinity)

# Add all variables to the model.
cpx = cplex.Cplex()
cpx.objective.set_sense(cpx.objective.sense.maximize)
# The CPLEX API expects a string for types. (e.g. "BBI..." etc.)
cpx.variables.add(obj=var_obj, lb=var_lb, ub=var_ub,
                  types="".join(var_types), names=var_names)

# =============================================================================
# ADDING CONSTRAINTS
# =============================================================================

# We now build a list of linear constraints.
cons_lin_expr = []  # list of [indices, coefficients]
cons_senses = []    # each element is "L" (<=), "G" (>=) or "E" (=)
cons_rhs = []       # right-hand side
cons_names = []     # names for each constraint

# --------------------------
# R1: For each l in L, sum_{j in J} y[j,l] <= n_HF[l]
# --------------------------
for l in L:
    indices = []
    coeffs = []
    for j in J:
        if (j, l) in y_vars:
            indices.append(y_vars[(j, l)])
            coeffs.append(1.0)
    cons_lin_expr.append([indices, coeffs])
    cons_senses.append("L")
    cons_rhs.append(n_HF[l])
    cons_names.append(f"R1_{l}")

# --------------------------
# R3: For each j in J, sum_{l in L} y[j,l] <= 1
# --------------------------
for j in J:
    indices = []
    coeffs = []
    for l in L:
        if (j, l) in y_vars:
            indices.append(y_vars[(j, l)])
            coeffs.append(1.0)
    cons_lin_expr.append([indices, coeffs])
    cons_senses.append("L")
    cons_rhs.append(1.0)
    cons_names.append(f"R3_{j}")

# --------------------------
# R4: For each i in I and each s in S, sum_{j in J_i[i]} x1[i,j,s] <= 1
# --------------------------
for i in I:
    if i in J_i and len(J_i[i]) > 0:
        for s in S:
            indices = []
            coeffs = []
            for j in J_i[i]:
                if (i, j, s) in x1_vars:
                    indices.append(x1_vars[(i, j, s)])
                    coeffs.append(1.0)
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("L")
            cons_rhs.append(1.0)
            cons_names.append(f"R4_{i}_{s}")

# --------------------------
# R6: For each i, j, and s1 != s2: x1[i,j,s2] - x1[i,j,s1] - z[j,s2] >= -1
# --------------------------
for i in I:
    for j in J:
        for s1 in S:
            for s2 in S:
                if s1 != s2:
                    indices = []
                    coeffs = []
                    if (i, j, s2) in x1_vars:
                        indices.append(x1_vars[(i, j, s2)])
                        coeffs.append(1.0)
                    else:
                        continue
                    if (i, j, s1) in x1_vars:
                        indices.append(x1_vars[(i, j, s1)])
                        coeffs.append(-1.0)
                    if (j, s2) in z_vars:
                        indices.append(z_vars[(j, s2)])
                        coeffs.append(-1.0)
                    cons_lin_expr.append([indices, coeffs])
                    cons_senses.append("G")
                    cons_rhs.append(-1.0)
                    cons_names.append(f"R6_{i}_{j}_{s1}_{s2}")

# --------------------------
# R7: For each i and for each s in S_l['hc'], sum_{j in J_HC} x2[i,j,s] == 1;
#      For s not in S_l['hc'], the sum equals 0.
# --------------------------
for i in I:
    for s in S:
        indices = []
        coeffs = []
        for j in J_HC:
            if (i, j, s) in x2_vars:
                indices.append(x2_vars[(i, j, s)])
                coeffs.append(1.0)
        if s in S_l.get("hc", set()):
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("E")
            cons_rhs.append(1.0)
            cons_names.append(f"R7_{i}_{s}")
        else:
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("E")
            cons_rhs.append(0.0)
            cons_names.append(f"R7_no_{i}_{s}")

# --------------------------
# R8: For each i, for each j in J_HC, and for s1 != s2: x2[i,j,s2] - x2[i,j,s1] - z[j,s2] >= -1
# --------------------------
for i in I:
    for j in J_HC:
        for s1 in S:
            for s2 in S:
                if s1 != s2:
                    indices = []
                    coeffs = []
                    if (i, j, s2) in x2_vars:
                        indices.append(x2_vars[(i, j, s2)])
                        coeffs.append(1.0)
                    else:
                        continue
                    if (i, j, s1) in x2_vars:
                        indices.append(x2_vars[(i, j, s1)])
                        coeffs.append(-1.0)
                    if (j, s2) in z_vars:
                        indices.append(z_vars[(j, s2)])
                        coeffs.append(-1.0)
                    cons_lin_expr.append([indices, coeffs])
                    cons_senses.append("G")
                    cons_rhs.append(-1.0)
                    cons_names.append(f"R8_{i}_{j}_{s1}_{s2}")

# --------------------------
# R9: For each i, for each j in J_HC, and for each s: x2[i,j,s] - x1[i,j,s] - y[j,'hc'] >= -1
# --------------------------
for i in I:
    for j in J_HC:
        for s in S:
            indices = []
            coeffs = []
            if (i, j, s) in x2_vars:
                indices.append(x2_vars[(i, j, s)])
                coeffs.append(1.0)
            else:
                continue
            if (i, j, s) in x1_vars:
                indices.append(x1_vars[(i, j, s)])
                coeffs.append(-1.0)
            if (j, "hc") in y_vars:
                indices.append(y_vars[(j, "hc")])
                coeffs.append(-1.0)
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("G")
            cons_rhs.append(-1.0)
            cons_names.append(f"R9_{i}_{j}_{s}")

# --------------------------
# R10: For each i, for each j in J_HC, for each s: t2max - t[i,j]*x2[i,j,s] >= 0
# --------------------------
for i in I:
    for j in J_HC:
        for s in S:
            indices = []
            coeffs = []
            indices.append(t2max_index)
            coeffs.append(1.0)
            if (i, j, s) in x2_vars:
                indices.append(x2_vars[(i, j, s)])
                coeffs.append(-t.loc[i, j])
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("G")
            cons_rhs.append(0.0)
            cons_names.append(f"R10_{i}_{j}_{s}")

# --------------------------
# R11: For each i, j in J, s in S: x1[i,j,s] <= sum_{l in L} y[j,l]
# --------------------------
for i in I:
    for j in J:
        for s in S:
            indices = []
            coeffs = []
            if (i, j, s) in x1_vars:
                indices.append(x1_vars[(i, j, s)])
                coeffs.append(1.0)
            else:
                continue
            for l in L:
                if (j, l) in y_vars:
                    indices.append(y_vars[(j, l)])
                    coeffs.append(-1.0)
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("L")
            cons_rhs.append(0.0)
            cons_names.append(f"R11_{i}_{j}_{s}")

# --------------------------
# R12: For each i, j in J_HC, s in S: x2[i,j,s] <= y[j,'hc']
# --------------------------
for i in I:
    for j in J_HC:
        for s in S:
            indices = []
            coeffs = []
            if (i, j, s) in x2_vars:
                indices.append(x2_vars[(i, j, s)])
                coeffs.append(1.0)
            else:
                continue
            if (j, "hc") in y_vars:
                indices.append(y_vars[(j, "hc")])
                coeffs.append(-1.0)
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("L")
            cons_rhs.append(0.0)
            cons_names.append(f"R12_{i}_{j}_{s}")

# --------------------------
# R13: For each j in J and each l in L: y[j,l] <= sum_{i in I, s in S} x1[i,j,s]
# --------------------------
for j in J:
    for l in L:
        indices = []
        coeffs = []
        if (j, l) in y_vars:
            indices.append(y_vars[(j, l)])
            coeffs.append(1.0)
        for i in I:
            for s in S:
                if (i, j, s) in x1_vars:
                    indices.append(x1_vars[(i, j, s)])
                    coeffs.append(-1.0)
        cons_lin_expr.append([indices, coeffs])
        cons_senses.append("L")
        cons_rhs.append(0.0)
        cons_names.append(f"R13_{j}_{l}")

# --------------------------
# R14: For each j in J_HC: y[j,'hc'] <= sum_{i in I, s in S} x2[i,j,s]
# --------------------------
for j in J_HC:
    indices = []
    coeffs = []
    if (j, "hc") in y_vars:
        indices.append(y_vars[(j, "hc")])
        coeffs.append(1.0)
    for i in I:
        for s in S:
            if (i, j, s) in x2_vars:
                indices.append(x2_vars[(i, j, s)])
                coeffs.append(-1.0)
    cons_lin_expr.append([indices, coeffs])
    cons_senses.append("L")
    cons_rhs.append(0.0)
    cons_names.append(f"R14_{j}")

# --------------------------
# R15: For each i, j in J, s in S: f1[i,j,s] - d1[i,s]*x1[i,j,s] = 0
# --------------------------
for i in I:
    for j in J:
        for s in S:
            indices = []
            coeffs = []
            if (i, j, s) in f1_vars:
                indices.append(f1_vars[(i, j, s)])
                coeffs.append(1.0)
            else:
                continue
            if (i, j, s) in x1_vars:
                indices.append(x1_vars[(i, j, s)])
                coeffs.append(-d1[(i, s)])
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("E")
            cons_rhs.append(0.0)
            cons_names.append(f"R15_{i}_{j}_{s}")

# --------------------------
# R16: For each i, j in J_HC, s in S: f2[i,j,s] - d2[i,s]*x2[i,j,s] = 0
# --------------------------
for i in I:
    for j in J_HC:
        for s in S:
            indices = []
            coeffs = []
            if (i, j, s) in f2_vars:
                indices.append(f2_vars[(i, j, s)])
                coeffs.append(1.0)
            else:
                continue
            if (i, j, s) in x2_vars:
                indices.append(x2_vars[(i, j, s)])
                coeffs.append(-d2[(i, s)])
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("E")
            cons_rhs.append(0.0)
            cons_names.append(f"R16_{i}_{j}_{s}")

# --------------------------
# R17: For each j in J_HP and each s in S:
#       sum_{i in I} f1[i,j,s] - (1/q[s])*sum_{p in P} c_work[p]*a_W[p,s]*w[j,p,s] - delta[j,s] <= 0
# --------------------------
for j in J_HP:
    for s in S:
        indices = []
        coeffs = []
        for i in I:
            if (i, j, s) in f1_vars:
                indices.append(f1_vars[(i, j, s)])
                coeffs.append(1.0)
        for p in P:
            if (j, p, s) in w_vars:
                indices.append(w_vars[(j, p, s)])
                coeffs.append(- (c_work[p] * a_W.get((p, s), 0) / q[s]))
        if (j, s) in delta_vars:
            indices.append(delta_vars[(j, s)])
            coeffs.append(-1.0)
        cons_lin_expr.append([indices, coeffs])
        cons_senses.append("L")
        cons_rhs.append(0.0)
        cons_names.append(f"R17_{j}_{s}")

# --------------------------
# R18: For each j in J_HC and each s in S:
#       sum_{i in I}(f1[i,j,s]+f2[i,j,s]) - (1/q[s])*sum_{p in P} c_work[p]*a_W[p,s]*w[j,p,s] - delta[j,s] <= 0
# --------------------------
for j in J_HC:
    for s in S:
        indices = []
        coeffs = []
        for i in I:
            if (i, j, s) in f1_vars:
                indices.append(f1_vars[(i, j, s)])
                coeffs.append(1.0)
            if (i, j, s) in f2_vars:
                indices.append(f2_vars[(i, j, s)])
                coeffs.append(1.0)
        for p in P:
            if (j, p, s) in w_vars:
                indices.append(w_vars[(j, p, s)])
                coeffs.append(- (c_work[p] * a_W.get((p, s), 0) / q[s]))
        if (j, s) in delta_vars:
            indices.append(delta_vars[(j, s)])
            coeffs.append(-1.0)
        cons_lin_expr.append([indices, coeffs])
        cons_senses.append("L")
        cons_rhs.append(0.0)
        cons_names.append(f"R18_{j}_{s}")

# --------------------------
# R18 (auxiliary): For each j in J and each s in S:
#       delta[j,s] - (sum_{i in I}(d1[i,s]+d2[i,s])) * z[j,s] <= 0
# --------------------------
for j in J:
    for s in S:
        total_demand = sum(d1[(i, s)] + d2[(i, s)] for i in I)
        indices = []
        coeffs = []
        if (j, s) in delta_vars:
            indices.append(delta_vars[(j, s)])
            coeffs.append(1.0)
        if (j, s) in z_vars:
            indices.append(z_vars[(j, s)])
            coeffs.append(-total_demand)
        cons_lin_expr.append([indices, coeffs])
        cons_senses.append("L")
        cons_rhs.append(0.0)
        cons_names.append(f"R18_aux_{j}_{s}")

# --------------------------
# R19: For each j in J: deltamax - sum_{s in S} delta[j,s] >= 0
# --------------------------
for j in J:
    indices = []
    coeffs = []
    for s in S:
        if (j, s) in delta_vars:
            indices.append(delta_vars[(j, s)])
            coeffs.append(-1.0)
    indices.append(deltamax_index)
    coeffs.append(1.0)
    cons_lin_expr.append([indices, coeffs])
    cons_senses.append("G")
    cons_rhs.append(0.0)
    cons_names.append(f"R19_{j}")

# --------------------------
# R20: For each j in J and each s in S: z[j,s] - sum_{l in L} a_HF[s,l]*y[j,l] <= 0
# --------------------------
for j in J:
    for s in S:
        indices = []
        coeffs = []
        if (j, s) in z_vars:
            indices.append(z_vars[(j, s)])
            coeffs.append(1.0)
        for l in L:
            if (j, l) in y_vars:
                indices.append(y_vars[(j, l)])
                coeffs.append(-a_HF.get((s, l), 0))
        cons_lin_expr.append([indices, coeffs])
        cons_senses.append("L")
        cons_rhs.append(0.0)
        cons_names.append(f"R20_{j}_{s}")

# --------------------------
# R21: For each j in J and each l in L: y[j,l] - sum_{s in S} z[j,s] <= 0
# --------------------------
for j in J:
    for l in L:
        indices = []
        coeffs = []
        if (j, l) in y_vars:
            indices.append(y_vars[(j, l)])
            coeffs.append(1.0)
        for s in S:
            if (j, s) in z_vars:
                indices.append(z_vars[(j, s)])
                coeffs.append(-1.0)
        cons_lin_expr.append([indices, coeffs])
        cons_senses.append("L")
        cons_rhs.append(0.0)
        cons_names.append(f"R21_{j}_{l}")

# --------------------------
# R22: For each j in J and each s in S: sum_{i in I} x1[i,j,s] - |I|*z[j,s] <= 0
# --------------------------
N = len(I)
for j in J:
    for s in S:
        indices = []
        coeffs = []
        for i in I:
            if (i, j, s) in x1_vars:
                indices.append(x1_vars[(i, j, s)])
                coeffs.append(1.0)
        if (j, s) in z_vars:
            indices.append(z_vars[(j, s)])
            coeffs.append(-N)
        cons_lin_expr.append([indices, coeffs])
        cons_senses.append("L")
        cons_rhs.append(0.0)
        cons_names.append(f"R22_{j}_{s}")

# --------------------------
# R23: For each j in J_HC and each s in S: sum_{i in I} x2[i,j,s] - |I|*z[j,s] <= 0
# --------------------------
for j in J_HC:
    for s in S:
        indices = []
        coeffs = []
        for i in I:
            if (i, j, s) in x2_vars:
                indices.append(x2_vars[(i, j, s)])
                coeffs.append(1.0)
        if (j, s) in z_vars:
            indices.append(z_vars[(j, s)])
            coeffs.append(-N)
        cons_lin_expr.append([indices, coeffs])
        cons_senses.append("L")
        cons_rhs.append(0.0)
        cons_names.append(f"R23_{j}_{s}")

# --------------------------
# R24: For each p in P: sum_{j in J, s in S} w[j,p,s] == n_W[p]
# --------------------------
for p in P:
    indices = []
    coeffs = []
    for j in J:
        for s in S:
            if (j, p, s) in w_vars:
                indices.append(w_vars[(j, p, s)])
                coeffs.append(1.0)
    cons_lin_expr.append([indices, coeffs])
    cons_senses.append("E")
    cons_rhs.append(n_W[p])
    cons_names.append(f"R24_{p}")

# --------------------------
# R25: For each j in J, p in P, s in S: w[j,p,s] - n_W[p]*a_W[p,s]*z[j,s] <= 0
# --------------------------
for j in J:
    for p in P:
        for s in S:
            indices = []
            coeffs = []
            if (j, p, s) in w_vars:
                indices.append(w_vars[(j, p, s)])
                coeffs.append(1.0)
            if (j, s) in z_vars:
                indices.append(z_vars[(j, s)])
                coeffs.append(-n_W[p] * a_W.get((p, s), 0))
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("L")
            cons_rhs.append(0.0)
            cons_names.append(f"R25_{j}_{p}_{s}")

# --------------------------
# R26: For each j in J, p in P, l in L: lb[p,l]*y[j,l] <= sum_{s in S} w[j,p,s]
# --------------------------
for j in J:
    for p in P:
        for l in L:
            indices = []
            coeffs = []
            if (j, l) in y_vars:
                indices.append(y_vars[(j, l)])
                coeffs.append(lb.get((p, l), 0))
            for s in S:
                if (j, p, s) in w_vars:
                    indices.append(w_vars[(j, p, s)])
                    coeffs.append(-1.0)
            cons_lin_expr.append([indices, coeffs])
            cons_senses.append("L")
            cons_rhs.append(0.0)
            cons_names.append(f"R26_{j}_{p}_{l}")

# --------------------------
# R27: For each j in J, s in S: z[j,s] - sum_{p in P} w[j,p,s] <= 0
# --------------------------
for j in J:
    for s in S:
        indices = []
        coeffs = []
        if (j, s) in z_vars:
            indices.append(z_vars[(j, s)])
            coeffs.append(1.0)
        for p in P:
            if (j, p, s) in w_vars:
                indices.append(w_vars[(j, p, s)])
                coeffs.append(-1.0)
        cons_lin_expr.append([indices, coeffs])
        cons_senses.append("L")
        cons_rhs.append(0.0)
        cons_names.append(f"R27_{j}_{s}")

# Add all constraints to the model.
cpx.linear_constraints.add(lin_expr=cons_lin_expr,
                           senses="".join(cons_senses),
                           rhs=cons_rhs,
                           names=cons_names)

# =============================================================================
# SOLVE THE MODEL
# =============================================================================

cpx.solve()
print(cpx.solution)

# =============================================================================
# DISPLAY SELECTED VARIABLES (only those with value > 0)
# =============================================================================

solution_vals = cpx.solution.get_values()

print("Selected variables (only those with positive values):")
for (j, l), idx in y_vars.items():
    if solution_vals[idx] > 0.5:
        print(f"y[{j},{l}] = {solution_vals[idx]}")
for (i, j, s), idx in x1_vars.items():
    if solution_vals[idx] > 0.5:
        print(f"x1[{i},{j},{s}] = {solution_vals[idx]}")
for (i, j, s), idx in x2_vars.items():
    if solution_vals[idx] > 0.5:
        print(f"x2[{i},{j},{s}] = {solution_vals[idx]}")
for (i, j, s), idx in f1_vars.items():
    if solution_vals[idx] > 0.0:
        print(f"f1[{i},{j},{s}] = {solution_vals[idx]}")
for (i, j, s), idx in f2_vars.items():
    if solution_vals[idx] > 0.0:
        print(f"f2[{i},{j},{s}] = {solution_vals[idx]}")
for (j, s), idx in z_vars.items():
    if solution_vals[idx] > 0.5:
        print(f"z[{j},{s}] = {solution_vals[idx]}")
for (j, p, s), idx in w_vars.items():
    if solution_vals[idx] > 0.0:
        print(f"w[{j},{p},{s}] = {solution_vals[idx]}")
if solution_vals[t2max_index] > 0.0:
    print(f"t2max = {solution_vals[t2max_index]}")
for (j, s), idx in delta_vars.items():
    if solution_vals[idx] > 0.0:
        print(f"delta[{j},{s}] = {solution_vals[idx]}")
if solution_vals[deltamax_index] > 0.0:
    print(f"deltamax = {solution_vals[deltamax_index]}")

# =============================================================================
# CREATE OUTPUT TABLE
# =============================================================================

table_data = []
for j in J:
    # Lost demand at facility j:
    lost_demand = sum(solution_vals[delta_vars[(j, s)]] for s in S if (j, s) in delta_vars)
    # Satisfied patients at j (based on workers allocated):
    satisfied_patients = 0.0
    for s in S:
        sum_workers = 0.0
        for p in P:
            if (j, p, s) in w_vars:
                sum_workers += c_work[p] * a_W.get((p, s), 0) * solution_vals[w_vars[(j, p, s)]]
        satisfied_patients += (1.0 / q[s]) * sum_workers
    # Total demand assigned to facility j:
    total_demand = 0.0
    if j in J_HC:
        for i in I:
            for s in S:
                if (i, j, s) in f1_vars and (i, j, s) in f2_vars:
                    total_demand += solution_vals[f1_vars[(i, j, s)]] + solution_vals[f2_vars[(i, j, s)]]
    else:
        for i in I:
            for s in S:
                if (i, j, s) in f1_vars:
                    total_demand += solution_vals[f1_vars[(i, j, s)]]
    table_data.append({
        "Facility": j,
        "Lost Demand": lost_demand,
        "Satisfied Patients": satisfied_patients,
        "Total Demand": total_demand
    })

output_table = pd.DataFrame(table_data)
print(output_table)

# =============================================================================
# PLOTTING THE SOLUTION
# =============================================================================

# Extract open facilities (those where any y[j,l] is 1)
open_facilities = {j: l for (j, l), idx in y_vars.items() if solution_vals[idx] > 0.5}
open_hfs_gdf = hfs_gdf[hfs_gdf['label'].isin(open_facilities.keys())].copy()
open_hfs_gdf['facility_type'] = open_hfs_gdf['label'].map(open_facilities)

# Extract first assignments (x1) and second assignments (x2)
assignments1 = {(i, j): 1 for (i, j, s), idx in x1_vars.items() if solution_vals[idx] > 0.5}
assignments2 = {(i, j): 1 for (i, j, s), idx in x2_vars.items() if solution_vals[idx] > 0.5}

# Build connection lists using the geometry from the GeoDataFrames.
connections1 = []
for (i, j) in assignments1.keys():
    start = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
    end = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]
    connections1.append((start, end))

connections2 = []
for (i, j) in assignments2.keys():
    start = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
    end = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]
    connections2.append((start, end))

# Plotting
fig, ax = plt.subplots(figsize=(5, 5))

# Plot demand points
demand_points_gdf.plot(ax=ax, color='red', label='Demand points')
for idx, row in demand_points_gdf.iterrows():
    ax.text(row.geometry.x + 0.05, row.geometry.y + 0.05, row['label'], fontsize=8, color='red')

# Plot open facilities: HPs and HCs
hps_open_gdf = open_hfs_gdf[open_hfs_gdf['facility_type'] == 'hp']
hcs_open_gdf = open_hfs_gdf[open_hfs_gdf['facility_type'] == 'hc']
hps_open_gdf.plot(ax=ax, color='blue', marker='^', label='HPs')
hcs_open_gdf.plot(ax=ax, color='black', marker='s', label='HCs')
for idx, row in open_hfs_gdf.iterrows():
    ax.text(row.geometry.x + 0.05, row.geometry.y + 0.05, row['label'], fontsize=8, color='black')

# Uncomment the following block to display first assignment connections:
# for (start, end) in connections1:
#     ax.annotate('', xy=(end.x, end.y), xytext=(start.x, start.y),
#                 arrowprops=dict(arrowstyle='->', color='gray', lw=1), zorder=1)

# Display second assignment connections
for (start, end) in connections2:
    ax.annotate('', xy=(end.x, end.y), xytext=(start.x, start.y),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1), zorder=1)

ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.5, 6.5)
ax.set_xticks([x * 0.5 for x in range(13)])
ax.set_yticks([y * 0.5 for y in range(13)])
ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

plt.show()
