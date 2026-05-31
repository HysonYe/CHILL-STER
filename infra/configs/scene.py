default_config = {
    "case": "default_case",
    "description": "Default simulation configuration. Since there are no targets, agents will not move under this configuration.",
    "environment": {
        "slot_size": 0.1,           # Slot duration in seconds
        "prop_speed": 1500,         # Speed of sound in water (m/s)
        "space":{                   # Dimensions of the simulation area (meters)
            "x": 1000,
            "y": 1000,
            "z": 100                # Water depth
        }
    },
    "nodes": [
        {
            "type": "TDMA",         # Protocol type
            "place": {              # Deployment strategy for nodes at different time slots
                "0": {              # Two deployment methods
                    "delay": 8,     # Specifies propagation delay; the environment generates a random position. If ≤ 0, the position is fully randomized within the area.
                    "coord": []     # Fixed coordinates for node deployment (highest priority)
                }
            },
            "strategy": {           # Time slot allocation strategy for nodes
                "0": {
                    "rand":[5,10],  # Randomized slot allocation: [number of occupied slots, frame size]
                    "fixed": [      # Static slot allocation; explicitly defines occupied slots (highest priority)
                                0,
                                0,
                                0,
                                0,
                                0,
                                1,
                                1,
                                1,
                                1,
                                1
                            ]
                }
            }
        }
    ],
    "agents": [
        {
            "type": "MobiU-MAC",
            "speed": 0.0,           # Agent movement speed (m/s)
                                    # When set to 0.0, the agent is considered stationary, and no delay estimation is performed.
            "place": {              # Deployment strategy for agents at different time slots
                "0": {
                    "delay": 5,
                    "coord": []
                }
            }
        }
    ],
    "targets":{                     # Waypoint targets configuration
        "padding": False,           # If True, randomly generate new coordinates to fill the target list once it has been fully traversed.
        "goals":[
                                    # List of target positions, each element is a tuple (x, y, z)
        ]
    }
}