# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

using KitePredictiveControl, KiteModels, KiteUtils

dt = 0.05
time = 10.0
vsm_dt = 0.1
plot_dt = 0.1

# Initialize model
set_data_path(joinpath(dirname(@__DIR__), "data"))
set_model = Settings("system_model.yaml")
model = SymbolicAWEModel(set_model)

KitePredictiveControl.linearize(model)

