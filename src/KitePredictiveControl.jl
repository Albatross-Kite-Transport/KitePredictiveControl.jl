# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

module KitePredictiveControl

using PrecompileTools
using SymbolicAWEModels
using ModelPredictiveControl
using ControlSystems
using Plots
using UnPack
using ModelingToolkit
import ModelingToolkit: t_nounits

@setup_workload begin
    @compile_workload begin
        @variables begin
            heading(t_nounits)[1]
            angle_of_attack(t_nounits)[1]
            tether_len(t_nounits)[1:3]
            winch_force(t_nounits)[1:3]
        end
        lin_outputs = [heading[1], angle_of_attack[1], tether_len[1], winch_force[1]]
        @info "Linear outputs: $lin_outputs"

        set = Settings("system.yaml")
        dt = 1/set.sample_freq
        sam = SymbolicAWEModel(set, "ram")
        init!(sam; lin_outputs)

        simple_plant_set = Settings("system.yaml")
        simple_plant_sam = SymbolicAWEModel(simple_plant_set, "simple_ram")
        init!(simple_plant_sam; lin_outputs)

        tether_set = Settings("system.yaml")
        tether_sam = SymbolicAWEModel(tether_set, "tether")
        init!(tether_sam)

        simple_set = Settings("system.yaml")
        simple_sam = SymbolicAWEModel(simple_set, "simple_ram")
        init!(simple_sam; lin_outputs)

        copy_to_simple!(sam, tether_sam, simple_sam; prn=false)
        copy_to_simple!(sam, tether_sam, simple_plant_sam; prn=false)
    end
end

end
