# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

"""
1. Speed/force controller for power tether
2. Length control for steering tethers
3. Linearize model:
    - left-right tether length difference vs heading
    - power-steering tether length difference vs angle of attack (or directly calculate the needed depower from geometry)
4. Find optimal tether lengths from linear model
5. Use winch controllers to go to these setpoints
"""