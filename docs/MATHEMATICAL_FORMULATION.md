# Mathematical Foundations of 3D Gaussian Splatting

This document outlines the complete mathematical framework implemented in Lumen for real-time 3D Gaussian radiance field synthesis, coordinate projections, and density optimization.

---

## 1. 3D Gaussian Representation

A 3D Gaussian primitive is parameterized by a spatial center position $\mu \in \mathbb{R}^3$, an anisotropic 3D scale vector $s = (s_x, s_y, s_z) \in \mathbb{R}^3_{>0}$, an orientation unit quaternion $q = (w, x, y, z) \in \mathbb{H}$ representing a rotation matrix $R \in \mathrm{SO}(3)$, an opacity scalar $\sigma \in [0, 1]$, and a set of spherical harmonics coefficients $k$.

The spatial probability density function of a 3D Gaussian is given by:

$$G(x) = \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

where $\Sigma \in \mathbb{R}^{3 \times 3}$ is the positive-definite spatial covariance matrix.

To guarantee that $\Sigma$ remains positive semi-definite throughout gradient optimization without expensive constrained projections, Lumen factorizes $\Sigma$ into scaling and rotation components:

$$\Sigma = R S S^T R^T$$

where $S = \operatorname{diag}(s_x, s_y, s_z)$ and $R$ is constructed from unit quaternion $q$:

$$R = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}$$

---

## 2. Projective Camera Transformation & Jacobian

Given a pinhole camera with intrinsic parameters focal lengths $(f_x, f_y)$, principal point $(c_x, c_y)$, and extrinsic rigid body transformation matrix $W \in \mathrm{SE}(3)$ mapping world coordinates to camera coordinates:

$$t = W \mu = \begin{bmatrix} t_x \\ t_y \\ t_z \end{bmatrix}$$

The non-linear perspective projection $\pi: \mathbb{R}^3 \to \mathbb{R}^2$ maps camera space coordinates to screen coordinates:

$$\pi(t) = \begin{bmatrix}
f_x \frac{t_x}{t_z} + c_x \\
f_y \frac{t_y}{t_z} + c_y
\end{bmatrix}$$

Because perspective projection is non-linear, projecting a 3D Gaussian directly does not yield an exact 2D Gaussian. Following Zwicker et al. (EWA Splatting), we compute the first-order Taylor series expansion (affine approximation) centered at the Gaussian position $t$:

$$J = \left. \frac{\partial \pi}{\partial t} \right|_t = \begin{bmatrix}
\frac{\partial \pi_x}{\partial t_x} & \frac{\partial \pi_x}{\partial t_y} & \frac{\partial \pi_x}{\partial t_z} \\
\frac{\partial \pi_y}{\partial t_x} & \frac{\partial \pi_y}{\partial t_y} & \frac{\partial \pi_y}{\partial t_z}
\end{bmatrix} = \begin{bmatrix}
\frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\
0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2}
\end{bmatrix}$$

---

## 3. Elliptical Weighted Average (EWA) Screen Covariance

Transforming the 3D covariance $\Sigma$ into camera coordinates produces $W \Sigma W^T$. Applying the projective Jacobian $J$ maps the covariance onto the 2D screen image plane:

$$\Sigma_{\text{screen}} = J W \Sigma W^T J^T$$

To prevent aliasing artifacts when a splat projects to an area smaller than a single pixel, an isotropic Gaussian reconstruction low-pass filter $\nu I_{2 \times 2}$ is added, where $\nu = 0.3 \text{ px}^2$:

$$\Sigma' = \Sigma_{\text{screen}} + \nu I_{2 \times 2} = \begin{bmatrix} \sigma_{xx}' & \sigma_{xy}' \\ \sigma_{xy}' & \sigma_{yy}' \end{bmatrix}$$

### 2D Footprint and Eigenvalue Decomposition

The screen extent of each Gaussian is bounded by a 3-sigma confidence ellipse containing 98.9% of its energy. The eigenvalues $\lambda_1, \lambda_2$ of $\Sigma'$ represent the semi-major and semi-minor axes:

$$\lambda_{1, 2} = \frac{\sigma_{xx}' + \sigma_{yy}'}{2} \pm \sqrt{\left(\frac{\sigma_{xx}' - \sigma_{yy}'}{2}\right)^2 + (\sigma_{xy}')^2}$$

The screen bounding radius $r_{\text{max}}$ is:

$$r_{\text{max}} = 3.0 \cdot \sqrt{\max(\lambda_1, \lambda_2)}$$

The bounding box on the screen is $[\lfloor x - r_{\text{max}} \rfloor, \lceil x + r_{\text{max}} \rceil] \times [\lfloor y - r_{\text{max}} \rfloor, \lceil y + r_{\text{max}} \rceil]$.

---

## 4. View-Dependent Color via Spherical Harmonics

View-dependent radiance is modeled using real spherical harmonics basis functions $Y_l^m(\theta, \phi)$ up to degree $l = 3$ (16 basis functions per color channel, yielding 48 scalar coefficients per Gaussian):

$$c(d) = \operatorname{clamp}\left(0.5 + \sum_{l=0}^3 \sum_{m=-l}^l k_l^m Y_l^m(d), 0, 1\right)$$

where $d = \frac{x - \mu}{\|x - \mu\|} \in S^2$ is the normalized viewing direction vector.

The 16 Cartesian basis functions implemented in Lumen are:

### Degree 0 ($l = 0$)
$$Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}} \approx 0.28209479$$

### Degree 1 ($l = 1$)
$$Y_1^{-1} = -\frac{1}{2} \sqrt{\frac{3}{\pi}} y \approx -0.48860251 y$$
$$Y_1^0 = \frac{1}{2} \sqrt{\frac{3}{\pi}} z \approx 0.48860251 z$$
$$Y_1^1 = -\frac{1}{2} \sqrt{\frac{3}{\pi}} x \approx -0.48860251 x$$

### Degree 2 ($l = 2$)
$$Y_2^{-2} = \frac{1}{2} \sqrt{\frac{15}{\pi}} x y \approx 1.09254843 x y$$
$$Y_2^{-1} = -\frac{1}{2} \sqrt{\frac{15}{\pi}} y z \approx -1.09254843 y z$$
$$Y_2^0 = \frac{1}{4} \sqrt{\frac{5}{\pi}} (2 z^2 - x^2 - y^2) \approx 0.31539156 (2 z^2 - x^2 - y^2)$$
$$Y_2^1 = -\frac{1}{2} \sqrt{\frac{15}{\pi}} x z \approx -1.09254843 x z$$
$$Y_2^2 = \frac{1}{4} \sqrt{\frac{15}{\pi}} (x^2 - y^2) \approx 0.54627421 (x^2 - y^2)$$

### Degree 3 ($l = 3$)
$$Y_3^{-3} = -\frac{1}{4} \sqrt{\frac{35}{\pi}} y (3x^2 - y^2) \approx -0.59004359 y (3x^2 - y^2)$$
$$Y_3^{-2} = \frac{1}{2} \sqrt{\frac{105}{\pi}} x y z \approx 2.89061144 x y z$$
$$Y_3^{-1} = -\frac{1}{4} \sqrt{\frac{21}{\pi}} y (4z^2 - x^2 - y^2) \approx -0.45704580 y (4z^2 - x^2 - y^2)$$
$$Y_3^0 = \frac{1}{4} \sqrt{\frac{7}{\pi}} z (2z^2 - 3x^2 - 3y^2) \approx 0.37317633 z (2z^2 - 3x^2 - 3y^2)$$
$$Y_3^1 = -\frac{1}{4} \sqrt{\frac{21}{\pi}} x (4z^2 - x^2 - y^2) \approx -0.45704580 x (4z^2 - x^2 - y^2)$$
$$Y_3^2 = \frac{1}{4} \sqrt{\frac{105}{\pi}} z (x^2 - y^2) \approx 1.44530572 z (x^2 - y^2)$$
$$Y_3^3 = -\frac{1}{4} \sqrt{\frac{35}{\pi}} x (x^2 - 3y^2) \approx -0.59004359 x (x^2 - 3y^2)$$

---

## 5. Volume Rendering & Front-to-Back Compositing

The radiance received at pixel $p = (u, v)$ is computed by integrating radiance along the viewing ray:

$$C(p) = \sum_{i \in \mathcal{N}} c_i \alpha_i T_i$$

where $\mathcal{N}$ represents the ordered set of Gaussians overlapping pixel $p$, sorted in ascending order of camera depth $t_{z, 1} \le t_{z, 2} \le \dots \le t_{z, N}$.

The effective opacity $\alpha_i$ of Gaussian $i$ at pixel coordinate $p$ is:

$$\alpha_i(p) = \sigma_i \cdot \exp\left(-\frac{1}{2} (p - \mu_i')^T (\Sigma_i')^{-1} (p - \mu_i')\right)$$

The transmittance $T_i$ represents the probability that the ray traverses from the camera through the first $i-1$ splats without absorption:

$$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$$

### Early Ray Termination

Because transmittance decreases monotonically as $T_{i+1} = T_i (1 - \alpha_i)$, once the accumulated opacity $A_i = 1 - T_{i+1}$ reaches the early termination threshold:

$$A_i \ge 0.99$$

subsequent Gaussians contribute less than 1% to the final pixel radiance. The pixel loop terminates immediately, saving memory bandwidth and arithmetic cycles.

---

## 6. Photometric Loss & Adaptive Density Control

The reconstruction loss combines L1 pixel deviation with the Structural Similarity Index Metric (SSIM):

$$\mathcal{L}_{\text{total}} = (1 - \lambda_{\text{ssim}}) \mathcal{L}_1 + \lambda_{\text{ssim}} (1 - \text{SSIM})$$

where standard tuning uses $\lambda_{\text{ssim}} = 0.2$.

### Adaptive Density Control (ADC) Rules

Throughout training iterations, view-space positional gradients are accumulated:

$$\nabla_{p} = \frac{1}{M} \sum_{m=1}^M \left\| \frac{\partial \mathcal{L}}{\partial \mu_{\text{screen}}} \right\|$$

1. **Under-reconstructed Regions** ($\nabla_p \ge \tau_{\text{pos}}$ and $\max(s) \le \tau_{\text{scale}}$):
   The Gaussian is cloned along the direction of its positional gradient.
2. **Over-reconstructed Regions** ($\nabla_p \ge \tau_{\text{pos}}$ and $\max(s) > \tau_{\text{scale}}$):
   The Gaussian is split into two smaller child Gaussians with scale reduced by a factor of 1.6 and positioned along the principal spatial axis.
3. **Pruning**:
   Gaussians with opacity $\sigma < 0.005$ or spatial extent exceeding the scene bounding volume are removed.
4. **Opacity Reset**:
   Every $N_{\text{reset}} = 3000$ iterations, opacities are reset to a small baseline value to purge redundant floater primitives.
