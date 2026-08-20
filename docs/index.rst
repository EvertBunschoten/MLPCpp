.. MLPToolbox documentation master file.

Welcome to MLPToolbox's Documentation!
======================================

MLPToolbox is designed to train Physics-Informed Neural Networks (PINNs) using 
reverse-mode Algorithmic Differentiation (AD) and features an advanced Gradient 
Annealing algorithm to mitigate gradient pathologies.

.. contents:: Table of Contents
   :depth: 3

Understanding Gradient Annealing & Lambda
========================================

In standard PINNs, the total loss is a composite of data-fit terms 
(:math:`L_i`, e.g., boundary conditions, reference data) and physics residual 
terms (:math:`L_r`, e.g., Navier-Stokes equations):

.. math::
   L(\theta) = L_r(\theta) + \sum_{i=1}^{M} \lambda_i L_i(\theta)

The Problem: Gradient Pathologies
--------------------------------
As discussed in *Wang et al. (2020)*, the back-propagated gradients of the PDE 
residual (:math:`\nabla_\theta L_r`) are often orders of magnitude larger than 
the data gradients (:math:`\nabla_\theta L_i`). This causes the optimizer to 
focus solely on minimizing the physics residual while ignoring the boundary/initial 
conditions, leading to erroneous predictions.

The Solution: Adaptive Lambda (:math:`\lambda_i`)
-------------------------------------------------
MLPToolbox implements **Paper Algorithm 1** to dynamically update :math:`\lambda_i` 
during training. Instead of manually tuning :math:`\lambda_i`, the trainer tracks 
gradient statistics:

1. **Instantaneous Lambda** (:math:`\hat{\lambda}_i`): Computed as the ratio 
   between the maximum absolute gradient of the physics residual and the mean 
   absolute gradient of the data term:
   
   .. math::
      \hat{\lambda}_i = \frac{\max_\theta |\nabla_\theta L_r|}{\overline{|\nabla_\theta L_i|}}

2. **Exponential Moving Average (EMA)**: To prevent oscillations, the actual 
   weight is smoothed using an EMA coefficient (:math:`\alpha`):
   
   .. math::
      \lambda_i \leftarrow (1 - \alpha)\lambda_i + \alpha \hat{\lambda}_i

**How it affects the loss:** By scaling :math:`L_i` by :math:`\lambda_i`, the 
effective learning rate of the data-fit terms is scaled up, forcing the network 
to respect boundary conditions and reference data even when the physics residuals 
are extremely stiff.

Core API Components
===================

MLPToolbox is modular. A typical workflow involves constructing a Network, 
defining Losses, configuring an Optimizer and Annealer, and passing them to 
the Trainer.

The Optimizer: CAdam
--------------------
A standard Adam optimizer implementation.

.. code-block:: cpp

   CAdam(
       mlpdouble learning_rate = 1e-3,
       mlpdouble beta1 = 0.9,
       mlpdouble beta2 = 0.999,
       mlpdouble epsilon = 1e-8
   )

The Annealer: CGradientAnnealer
-----------------------------------
The annealer automatically balances the loss terms. It is configured using 
the ``AnnealerConfig`` struct.

**AnnealerConfig Parameters:**

* ``alpha``: Default `0.9`. Controls the EMA smoothing. A higher value makes 
  :math:`\lambda` react faster to gradient changes, while a lower value smooths 
  it out. `0.9` is recommended by the paper.
* ``lambda_init``: Default `1.0`. The initial :math:`\lambda` for all data terms.
* ``lambda_max``: Default `1e4`. **Crucial parameter**. If your non-dimensionalization 
  introduces large scaling coefficients, the physics gradients will be massive, 
  driving :math:`\hat{\lambda}_i` to infinity. The hard clamp prevents gradient explosion.
* ``lambda_min``: Default `1e-4`. Prevents a data term from being completely ignored.
* ``n_data_terms``: Automatically calculated by the ``CMLPTrainer`` during 
  ``Build()`` as `n_reference_losses + n_boundary_losses`.

The Trainer: CMLPTrainer
------------------------
The trainer orchestrates the AD tape, mini-batching, MPI synchronization, and 
the fast/slow path gradient evaluations.

**TrainerConfig Parameters:**

* ``batch_size``: The number of reference data points processed per gradient step.
* ``physics_batch_size``: The number of collocation points processed per step. 
  **Performance Note:** Computing Hessians (2nd derivatives) for momentum equations 
  is memory and CPU intensive. Keep this between 32 and 128.
* ``use_annealer``: Set to `true` to enable the gradient pathology mitigation.
* ``annealer_update_freq``: **Performance Note**. Computing :math:`\hat{\lambda}_i` 
  requires :math:`N+1` reverse sweeps of the AD tape. Setting this to `10` means 
  the expensive "Slow Path" only runs every 10 steps. On the other 9 steps, the 
  trainer uses a "Fast Path" (1 sweep) with the cached :math:`\lambda` values.
* ``max_grad_norm``: If > 0.0, applies global gradient clipping to prevent NaNs.

Example: 2D Lid-Driven Cavity
=============================

This section provides a comprehensive, section-by-section walkthrough of the 
``cavity_trainer.cpp`` code. It explains the physics, the non-dimensionalization, 
and how the MLPToolbox API is used to implement Paper Algorithm 1 for the 2D 
Lid-Driven Cavity flow.

1. MPI Initialization & Physical Scales
--------------------------------------
First, we initialize MPI and define the physical reference scales of the problem. 
The cavity is a :math:`1 \times 1` box (:math:`L_{ref}=1`). The lid moves at 
:math:`U_{lid} = 33.179` m/s.

.. code-block:: cpp

   MPI_Init(&argc, &argv);
   int rank = 0, size = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   const double U_lid = 33.179;
   const double L_ref = 1.0;
   const double rho_ref = 5.04175e-04;
   const double R_gas = 287.058; 
   const double T_ref = 288.15;
   const double p_ref = rho_ref * R_gas * T_ref; 
   const double C_p = (R_gas * T_ref) / (U_lid * U_lid); 
   const double Re = 2000.0; 

**Key Physics Note (Pressure Scaling):**
Instead of scaling pressure by dynamic pressure (:math:`\rho U^2`), we scale it by 
ideal gas static pressure (:math:`\rho R T`). Because of this specific pressure scaling, 
the pressure gradient term in the Navier-Stokes equations needs a coefficient: 
:math:`C_p = \frac{R T}{U_{lid}^2} \approx 75`. The Reynolds number is set to 2000.

2. Data Loading & Non-Dimensionalization
----------------------------------------
The ``ReadSU2CSV_NonDim`` function reads a SU2 CFD restart file. SU2 outputs 
conservative variables (Density, Momentum, Energy). The function converts these to 
primitives (:math:`u, v, p`), and then **non-dimensionalizes** them so the neural 
network trains on values that are :math:`\mathcal{O}(1)`.

.. code-block:: cpp

   Y.push_back({
       mlpdouble(u_dim / U_ref), 
       mlpdouble(v_dim / U_ref), 
       mlpdouble(p_dim / p_ref)
   });

3. Neural Network Architecture
------------------------------
A simple Multi-Layer Perceptron (MLP) is used: 2 inputs :math:`(x,y)`, three hidden 
layers of 32 neurons (Tanh activation), and 3 outputs :math:`(u,v,p)` (Linear activation). 
Crucially, min-max normalization is applied to keep the internal math of the network stable.

.. code-block:: cpp

   std::vector<std::size_t> architecture = {2, 32, 32, 32, 3};
   CNeuralNetwork net(architecture);
   for (std::size_t iLayer = 1; iLayer < net.GetnLayers() - 1; ++iLayer) {
       net.SetActivationFunction(iLayer, "tanh");
   }
   net.SetActivationFunction(net.GetnLayers() - 1, "linear");

   net.SetInputRegularization("minmax");
   net.SetInputNorm(0, 0.0, 1.0); // x in [0,1]
   net.SetInputNorm(1, 0.0, 1.0); // y in [0,1]

   net.SetOutputRegularization("minmax");
   net.SetOutputNorm(0, -1.0, 1.0); // u, v, p normalized to [-1, 1]
   net.SetOutputNorm(1, -1.0, 1.0);
   net.SetOutputNorm(2, -1.0, 1.0);

4. Physics Residuals (The PDEs, :math:`L_r`)
---------------------------------------------
This section defines the residuals. The trainer will enforce that these residuals 
equal zero at collocation points inside the domain.

**Signed Distance Function (SDF) Weighting:**
To prevent the stiff boundary layers from causing gradient explosions early in 
training, we down-weight the physics near the walls using an SDF.

.. code-block:: cpp

   auto sdf_weight = [](const PhysicsState& s) -> double {
       double x = to_double(s.In(0)); double y = to_double(s.In(1));
       return std::min({x, 1.0 - x, y, 1.0 - y});
   };

**Continuity Equation:**
Enforces incompressibility (:math:`\nabla \cdot \mathbf{u} = 0`). ``s.Jac(output, input)`` 
provides the Jacobian.

.. code-block:: cpp

   eq_cont.residual = [sdf_weight](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
       mlpdouble res = s.Jac(0, 0) + s.Jac(1, 1); // du/dx + dv/dy
       return res * mlpdouble(sdf_weight(s));
   };

**X-Momentum Equation:**
The steady-state x-momentum Navier-Stokes equation. Note the use of ``s.Hess(output, input_i, input_j)`` 
to get the 2nd derivatives (Laplacian) for the viscous terms, and the inclusion of the 
:math:`C_p` coefficient for the pressure gradient.

.. code-block:: cpp

   eq_xmom.residual = [Re, C_p, sdf_weight](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
       mlpdouble u = s.Out(0);
       mlpdouble v = s.Out(1);
       mlpdouble dudx = s.Jac(0, 0);
       mlpdouble dudy = s.Jac(0, 1);
       mlpdouble dpdx = s.Jac(2, 0);
       mlpdouble d2udx2 = s.Hess(0, 0, 0);
       mlpdouble d2udy2 = s.Hess(0, 1, 1);
       mlpdouble res = u * dudx + v * dudy + C_p * dpdx - (1.0 / Re) * (d2udx2 + d2udy2);
       return res * mlpdouble(sdf_weight(s));
   };

5. Boundary Condition Losses (:math:`L_i`)
-----------------------------------------
Boundary conditions are treated as physics losses evaluated on boundary collocation sets.

**Corner Smoothing for Top Wall:**
At the top wall (:math:`y=1`), :math:`u=1`. However, at the top corners (:math:`x=0` and :math:`x=1`), 
there is a mathematical discontinuity (the lid moves, but the side walls are fixed). This logic 
linearly ramps :math:`u` from 0 to 1 near the corners, preventing the network from trying to learn 
an infinite gradient.

.. code-block:: cpp

   eq_top_u.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
       double x = to_double(s.In(0));
       double target_u = 1.0;
       if (x < 0.1) target_u = x / 0.1;
       else if (x > 0.9) target_u = (1.0 - x) / 0.1;
       return s.Out(0) - mlpdouble(target_u);
   };

6. Optimizer, Annealer, and Trainer Configuration
-------------------------------------------------
The Annealer is configured exactly as the paper recommends (:math:`\alpha=0.9`). 
The ``lambda_max=1e4`` clamps the weights because the :math:`C_p=75` makes the physics 
gradients massive. The trainer uses ``annealer_update_freq=10`` to run the "Slow Path" 
every 10 steps, and the "Fast Path" for the other 9.

.. code-block:: cpp

   CAdam optimizer(1e-3, 0.9, 0.999, 1e-8);
   AnnealerConfig anneal_cfg;
   anneal_cfg.alpha = 0.9; 
   anneal_cfg.lambda_max = 1e4;

   TrainerConfig trainer_cfg;
   trainer_cfg.max_epochs         = 500;
   trainer_cfg.batch_size         = 128;    
   trainer_cfg.physics_batch_size = 32;  // Keep small for Hessian memory
   trainer_cfg.use_annealer       = true;
   trainer_cfg.annealer_update_freq = 10;
   trainer_cfg.shuffle_per_epoch  = true;

   CMLPTrainer trainer(net, std::move(optimizer), anneal_cfg, trainer_cfg);

7. Data Partitioning & Collocation Points (MPI)
-----------------------------------------------
The domain is sampled randomly. With 2 MPI ranks, each rank gets 2500 interior points, 
500 top boundary points, and 1500 wall boundary points. The trainer is told which points 
belong to which named set (``"interior"``, ``"top_bc"``, ``"wall_bc"``).

.. code-block:: cpp

   std::size_t local_coll = (5000 + size - 1) / size;
   for (std::size_t i = 0; i < local_coll; ++i) {
       physics_points.push_back({mlpdouble(dist(gen)), mlpdouble(dist(gen))});
   }
   trainer.SetCollocationPoints("interior", physics_points);
   // ... generate boundary points ...

8. Registering Losses
---------------------
The separation of API calls maps directly to Paper Algorithm 1:

1. **Reference Data** (:math:`L_{data}`) and **Boundary Conditions** (:math:`L_{bc}`) are 
   registered as :math:`L_i` terms. The annealer will assign a :math:`\lambda` weight to each.
2. **Physics Losses** (:math:`L_{phys}`) are registered as :math:`L_r` terms. They always have a weight of 1.0.

.. code-block:: cpp

   trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>()); // SU2 Data
   
   trainer.AddPhysicsLoss(loss_cont, "interior");
   trainer.AddPhysicsLoss(loss_xmom, "interior");
   trainer.AddPhysicsLoss(loss_ymom, "interior");
   
   trainer.AddBoundaryLoss(loss_top_u, "top_bc");
   trainer.AddBoundaryLoss(loss_top_v, "top_bc");
   trainer.AddBoundaryLoss(loss_wall_u, "wall_bc");
   trainer.AddBoundaryLoss(loss_wall_v, "wall_bc");

9. The Training Loop & Logging
------------------------------
``TrainEpoch()`` handles everything internally: mini-batching, AD tape registration, the 
Fast/Slow gradient sweeps, MPI ``Allreduce`` synchronization, and the Adam weight update. 
The user only needs to poll the trainer for the average losses and the average :math:`\lambda` 
to log the training history.

.. code-block:: cpp

   for (std::size_t epoch = 0; epoch < trainer_cfg.max_epochs; ++epoch) {
       trainer.TrainEpoch();
       
       const double avg_data_loss = trainer.GetEpochAverageLossRef();
       double avg_physics_loss = 0.0;
       for (size_t k = 0; k < 3; ++k) {
           avg_physics_loss += trainer.GetEpochAverageLossPhys(k);
       }
       double avg_bc_loss = 0.0;
       for (size_t k = 0; k < 4; ++k) {
           avg_bc_loss += trainer.GetEpochAverageLossBC(k);
       }
       const double avg_lambda    = trainer.GetEpochAverageLambda(0);
       const double avg_loss_total = trainer.GetEpochAverageLossTotal();
       
       // ... log to CSV and console ...
   }

10. Output Predictions (Dimensionalization)
-------------------------------------------
After training, the network predicts non-dimensional variables. To visualize the flow 
in standard tools (like ParaView), we evaluate the network on a dense :math:`100 \times 100` 
grid and multiply the outputs back by the physical reference scales (:math:`U_{lid}` and :math:`p_{ref}`) 
to restore their dimensional values.

.. code-block:: cpp

   for (int i = 0; i <= 100; ++i) {
       for (int j = 0; j <= 100; ++j) {
           double x = static_cast<double>(i) / 100.0;
           double y = static_cast<double>(j) / 100.0;
           std::vector<mlpdouble> input = {mlpdouble(x), mlpdouble(y)};
           
           net.Predict(input, false, false);
           
           double u_dim = to_double(net.GetOutput(0)) * U_lid;
           double v_dim = to_double(net.GetOutput(1)) * U_lid;
           double p_dim = to_double(net.GetOutput(2)) * p_ref;
           
           pred_file << x << "," << y << "," << u_dim << "," << v_dim << "," << p_dim << "\n";
       }
   }