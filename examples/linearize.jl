# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

using KitePredictiveControl, KiteModels, ControlPlots, SparseArrays

"""
Plot the linearization error over time at [0, 1, 5] seconds into the future of all outputs.


"""


# julia> A[2]
# 0.43297457652132393

# julia> X_prime[1]
# 9.839349596666946

# size(X) = (nm, nsx)
# size(U) = (nm, nu)
# size(X_prime) = (nm, nsx)
# X_prime' = [A B] * [X' ; U']