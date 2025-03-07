# scenarios.py

scenarios = {
    "baseline": {
        "location_file": "location_refcamps.geojson",
        "HFs_to_locate": [2, 3],
        "t1max": 3.5,
        "workers_to_allocate": [3, 8, 8],  # doctor, nurse, midwife
        "total_population": 16045,
        "demand_rate_opening_hours": [0.00835, 0.00162, 0.00010],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.03478, 0.00175, 0.00069],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "ub_workers": [
            [3, 3],  # doctor
            [8, 8],  # nurse
            [8, 8],  # midwife
        ],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
        "baseline2": {
        "location_file": "location_refcamps2.geojson",
        "distance_matrix": "distance_matrix_refcamps_meters.xlsx",
        "HFs_to_locate": [6, 8],
        "t1max": 1000, # meters (based on avg distance from demand points to HPs candidate locations)
        "t2max": 4500, # meters (based on avg distance from demand points to HCs candidate locations and also min intercamp distance)
        "workers_to_allocate": [6, 20, 20],  # doctor, nurse, midwife 
        "total_population": 16045,
        "demand_rate_opening_hours": [0.00835, 0.00162, 0.00010],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.03478, 0.00175, 0.00069],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "ub_workers": [
            [6, 20],  # doctor
            [6, 20],  # nurse
            [6, 20],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
        "tierkidi": {
        "location_file": "tierkidi.geojson",
        "distance_matrix": "distance_matrix_tierkidi.xlsx",
        "HFs_to_locate": [2, 1],
        "t1max": 1000, # meters (based on avg distance from demand points to HPs candidate locations)
        "t2max": 3500, # meters (based on avg distance from demand points to HCs candidate locations)
        "workers_to_allocate": [1, 14, 14],  # doctor, nurse, midwife 
        "total_population": 700,
        "demand_rate_opening_hours": [0.00835, 0.00162, 0.00010],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.03478, 0.00175, 0.00069],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "ub_workers": [
            [1, 1],  # doctor
            [14, 14],  # nurse
            [14, 14],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
        "terkidi_baseline": {
        "location_file": "terkidi_baseline.geojson",
        "distance_matrix": "distance_matrix_terkidi_baseline.xlsx",
        "HFs_to_locate": [2, 1],
        "t1max": 1000, # meters (based on avg distance from demand points to HPs candidate locations)
        "t2max": 3500, # meters (based on avg distance from demand points to HCs candidate locations)
        "workers_to_allocate": [1, 14, 14],  # doctor, nurse, midwife 
        "total_population": 700,
        "demand_rate_opening_hours": [0.01428571429, 0.00428571428, 0.00142857142],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.00428571428, 0.00285714285, 0.00142857142],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "ub_workers": [
            [1, 1],  # doctor
            [14, 14],  # nurse
            [14, 14],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
    "terkidi_baseline_change_t1max": {
        "location_file": "terkidi_baseline.geojson",
        "distance_matrix": "distance_matrix_terkidi_baseline.xlsx",
        "HFs_to_locate": [2, 1],
        "t1max": 500, # meters (based on avg distance from demand points to HPs candidate locations)
        "t2max": 3500, # meters (based on avg distance from demand points to HCs candidate locations)
        "workers_to_allocate": [1, 14, 14],  # doctor, nurse, midwife 
        "total_population": 700,
        "demand_rate_opening_hours": [0.01428571429, 0.00428571428, 0.00142857142],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.00428571428, 0.00285714285, 0.00142857142],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "ub_workers": [
            [1, 1],  # doctor
            [14, 14],  # nurse
            [14, 14],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
        "terkidi_add_1HC": {
        "location_file": "terkidi_baseline.geojson",
        "distance_matrix": "distance_matrix_terkidi_baseline.xlsx",
        "HFs_to_locate": [2, 2],
        "t1max": 1000, # meters (based on avg distance from demand points to HPs candidate locations)
        "t2max": 3500, # meters (based on avg distance from demand points to HCs candidate locations)
        "workers_to_allocate": [1, 14, 14],  # doctor, nurse, midwife 
        "total_population": 700,
        "demand_rate_opening_hours": [0.01428571429, 0.00428571428, 0.00142857142],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.00428571428, 0.00285714285, 0.00142857142],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "ub_workers": [
            [1, 1],  # doctor
            [14, 14],  # nurse
            [14, 14],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
        "terkidi_add_1HC_add_1doctor": {
        "location_file": "terkidi_baseline.geojson",
        "distance_matrix": "distance_matrix_terkidi_baseline.xlsx",
        "HFs_to_locate": [2, 2],
        "t1max": 1000, # meters (based on avg distance from demand points to HPs candidate locations)
        "t2max": 3500, # meters (based on avg distance from demand points to HCs candidate locations)
        "workers_to_allocate": [2, 14, 14],  # doctor, nurse, midwife 
        "total_population": 700,
        "demand_rate_opening_hours": [0.01428571429, 0.00428571428, 0.00142857142],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.00428571428, 0.00285714285, 0.00142857142],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "ub_workers": [
            [1, 1],  # doctor
            [14, 14],  # nurse
            [14, 14],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
        "terkidi_add_1HC_add_1doctor_change_t1max": {
        "location_file": "terkidi_baseline.geojson",
        "distance_matrix": "distance_matrix_terkidi_baseline.xlsx",
        "HFs_to_locate": [2, 2],
        "t1max": 500, # meters (based on avg distance from demand points to HPs candidate locations)
        "t2max": 3500, # meters (based on avg distance from demand points to HCs candidate locations)
        "workers_to_allocate": [2, 14, 14],  # doctor, nurse, midwife 
        "total_population": 700,
        "demand_rate_opening_hours": [0.01428571429, 0.00428571428, 0.00142857142],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.00428571428, 0.00285714285, 0.00142857142],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "ub_workers": [
            [2, 2],  # doctor
            [14, 14],  # nurse
            [14, 14],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
        "terkidi_add_1HC_add_1doctor_change_t1max_add_4nurses4midwives": {
        "location_file": "terkidi_baseline.geojson",
        "distance_matrix": "distance_matrix_terkidi_baseline.xlsx",
        "HFs_to_locate": [2, 2],
        "t1max": 500, # meters (based on avg distance from demand points to HPs candidate locations)
        "t2max": 3500, # meters (based on avg distance from demand points to HCs candidate locations)
        "workers_to_allocate": [2, 18, 18],  # doctor, nurse, midwife 
        "total_population": 700,
        "demand_rate_opening_hours": [0.01428571429, 0.00428571428, 0.00142857142],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.00428571428, 0.00285714285, 0.00142857142],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "ub_workers": [
            [2, 2],  # doctor
            [18, 18],  # nurse
            [18, 18],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
        "terkidi_add_1HC_add_1doctor_change_t1max_add_4nurses4midwives_upper_bounds": {
        "location_file": "terkidi_baseline.geojson",
        "distance_matrix": "distance_matrix_terkidi_baseline.xlsx",
        "HFs_to_locate": [2, 2],
        "t1max": 500, # meters (based on avg distance from demand points to HPs candidate locations)
        "t2max": 3500, # meters (based on avg distance from demand points to HCs candidate locations)
        "workers_to_allocate": [2, 18, 18],  # doctor, nurse, midwife 
        "total_population": 700,
        "demand_rate_opening_hours": [0.01428571429, 0.00428571428, 0.00142857142],  # basic, maternal1, maternal2
        "demand_rate_closing_hours": [0.00428571428, 0.00285714285, 0.00142857142],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [0, 1],  # doctor
            [1, 2],  # nurse
            [1, 2],  # midwife
        ],
        "ub_workers": [
            [1, 2],  # doctor
            [6, 8],  # nurse
            [6, 8],  # midwife
        ],
        "services_at_HFs": [
            [1, 0],  # basic
            [1, 1],  # maternal1
            [0, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 0, 1],  # doctor
            [1, 1, 0],  # nurse
            [0, 1, 1],  # midwife
        ],
    },
    "high_demand": {
        "location_file": "location_refcamps.geojson",
        "HFs_to_locate": [3, 5],
        "t1max": 4.0,
        "workers_to_allocate": [5, 10, 10],
        "total_population": 20000,
        "demand_rate_opening_hours": [0.009, 0.002, 0.00015],
        "demand_rate_closing_hours": [0.040, 0.002, 0.001],
        "working_hours": [7, 8, 8],
        "service_time": [0.5, 1, 2],
        "lb_workers": [
            [1, 2],  # doctor
            [2, 3],  # nurse
            [2, 3],  # midwife
        ],
        "services_at_HFs": [
            [1, 1],  # basic
            [1, 1],  # maternal1
            [1, 1],  # maternal2
        ],
        "services_per_worker": [
            [1, 1, 1],  # doctor
            [1, 1, 1],  # nurse
            [1, 1, 1],  # midwife
        ],
    },
}
