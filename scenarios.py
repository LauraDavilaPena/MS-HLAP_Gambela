# scenarios.py

scenarios = {
    "baseline": {
        "HFs_to_locate": [2, 3],
        "t1max": 3.5,
        "workers_to_allocate": [3, 8, 8],  # doctor, nurse, midwife
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
