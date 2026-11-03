2. A Physics-Informed Machine Learning Framework for Predicting Transient Temperature Distribution in Moving Heat Source Welding Processes
2.1 Introduction: 
Gas Metal Arc Welding (GMAW) is a widely used fusion-welding process in which an electric arc generates the thermal energy required to melt and join metallic components. During welding, the arc travels along a predefined path, continuously transferring heat to the workpiece and producing a spatially and temporally varying temperature field. The movement of the heat source results in rapid heating near the welding arc, followed by progressive cooling of the material behind it, leading to steep thermal gradients and complex temperature histories across the plate. Accurate prediction of these transient temperature distributions is essential for understanding the thermal response of welded structures, particularly the development of the heat-affected zone, microstructural changes, residual stresses, and distortion. Conventional stationary or uniformly distributed heat-source representations may not adequately capture the localized and asymmetric thermal energy distribution associated with a travelling welding arc. Therefore, an appropriate moving heat-source model is necessary to represent the spatial distribution and time-dependent movement of heat during the welding process.
     In the present investigation, a three-dimensional transient thermal simulation framework is developed for GMAW of API 5L X80 steel using the Goldak double-ellipsoidal moving heat-source model. This model represents the welding arc as two adjoining volumetric ellipsoids corresponding to the front and rear regions of the heat source, allowing their respective dimensions and heat-distribution fractions to be defined independently. The instantaneous heat-source position is updated with time according to the welding travel speed, thereby reproducing the progressive movement of the arc along the prescribed weld path. The resulting volumetric heat-generation term is incorporated into the transient heat-conduction equation, considering temperature-dependent thermal properties and the relevant thermal boundary conditions. The simulation is used to predict temperature histories at selected locations across the plate, including different transverse offsets from the weld centreline and depths through the plate thickness. The predicted peak temperatures are subsequently compared with reference (Rocha et al. 2018) finite element and infrared thermography results to assess the thermal response captured by the numerical model. This physics-based simulation also establishes a foundation for the subsequent development of a physics-informed machine-learning framework for predicting transient temperature distributions under varying welding conditions.




2.2. Experimental background (Rocha et al. 2018):
 
Figure 1. Schematic representation of the GMAW process showing the API 5L X80 steel workpiece geometry, moving welding torch, weld path, coordinate system, thermal boundary conditions, and temperature measurement locations. All dimensions material properties, welding conditions were obtained from Rocha et al. 2018.
Table 1. Welding conditions, workpiece geometry and heat-source parameters obtained from Rocha et al. 2018
	Parameter	Value
Welding process
	Welding process	GMAW
	Workpiece material	API 5L X80 steel
	Welding configuration	Single weld bead
Welding parameters	Welding current (I)	192 A
	Arc voltage (V)	18.4 V
	Thermal efficiency (η)	0.85
	Gross electrical power (VI)	3532.8 W
	Net heat input power (Q = ηVI)	3002.88 W
Welding motion	Welding travel speed (v)	2.78 mm/s
	Welding direction	+x
	Approximate welding duration	35.97 s
Goldak heat-source model
	Heat-source formulation	Double ellipsoidal
	Front longitudinal parameter (af)	3.025 mm
	Rear longitudinal parameter (ar)	12.100 mm
	Transverse parameter (b)	4.472 mm
	Depth parameter (c)	6.050 mm
	Front heat-distribution factor (ff)	0.4
	Rear heat-distribution factor (fr)	1.6
Table 2. Temperature dependent thermo-physical properties obtained from Rocha et al. 2018
Temperature (°C)	Thermal Conductivity (W/m°C)	Specific heat Capacity (J/kg°C)	Density (kg/m3)
0	51.062	571.235	6384.912889
100	51.062	571.235	6797.62589
200	47.324	571.235	6915.280782
300	43.84	571.235	7060.349375
400	40.424	614.841	6799.07749
500	36.95	668.617	6709.157113
550	33.546	730.831	6238.267569
600	26.461	780.774	5656.940234
650	18.019	868.623	5688.052513














2.3 Loading and Boundary Conditions: 
 
Figure 2. Schematic representation of the moving Goldak heat-source loading and thermal boundary conditions applied in the three-dimensional transient GMAW simulation of API 5L X80 steel.
2.4 3D cell-centred, explicit finite-volume method:
The three-dimensional transient heat-conduction equation governing the welding process is solved using a cell-centred explicit finite-volume numerical approach. The computational domain representing the API 5L X80 steel plate is discretised into a structured Cartesian grid consisting of 50, 80, and 5 cells along the longitudinal (x), transverse (y), and thickness (z) directions, respectively. The corresponding spatial resolutions are 3 mm, 1 mm, and 1.5 mm, resulting in a total of 20,000 computational cells. The temperature is evaluated at the centre of each computational cell, and the thermophysical properties, including thermal conductivity, specific heat capacity, and density, are updated as functions of the local temperature using piecewise-linear interpolation of the supplied material-property data.
      The governing heat-conduction equation is discretised spatially by evaluating the conductive heat fluxes across the cell faces. Harmonic averaging is employed to determine the effective thermal conductivity at the interfaces between adjacent computational cells, thereby accounting for temperature-dependent conductivity variations across the domain. The resulting conductive heat-flux divergence is combined with the volumetric heat-generation term from the moving Goldak double-ellipsoidal heat source and the heat losses associated with the prescribed thermal boundary conditions. Temporal integration is performed using the forward Euler explicit scheme, in which the temperature at the next time step is calculated from the temperature and heat-transfer contributions at the current time step. The time increment is restricted by the explicit stability criterion and an additional source-heating constraint to control the temperature rise per computational step. For the present simulation, a time increment of approximately (8.1055×10^-4) s is employed, resulting in 44,379 computational time steps over the welding duration. The numerical implementation also incorporates global energy accounting by tracking the cumulative heat input, boundary heat losses, and change in sensible thermal energy. This provides a quantitative assessment of the discrete energy conservation of the numerical solution.
The temperature update is performed as:
T_P^(n+1)=T_P^n+Δt/(ρ_P^n c_(p,P)^n ) [∇⋅(k∇T)_P^n+q ̇_P^n-Q ̇_(loss,P)^n/V_P ]
where n and n+1 denote the current and subsequent time levels, respectively, and Δt is the computational time increment.
Table 3. Numerical solver configuration, computational grid parameters, temporal discretisation, thermal boundary conditions, and temperature extraction procedures employed in the three-dimensional transient GMAW thermal simulation.
Parameter	Description
Spatial discretisation	Cell-centred finite-volume formulation
Computational geometry	3D Cartesian coordinate system
Temporal integration	First-order forward Euler explicit scheme
Number of cells along x-direction	50
Number of cells along y-direction	80
Number of cells along z-direction	5
Total computational cells	20,000
Grid spacing, Δx	3.0 mm
Grid spacing, Δy	1.0 mm
Grid spacing, Δz	1.5 mm
Selected time increment, Δt	8.10553 × 10⁻⁴ s
Diffusion stability limit	8.03571 × 10⁻³ s
Total computational time steps	44,379
Total simulated welding time	Approximately 35.97 s
Internal face conductivity	Harmonic-mean averaging
Top and four side surfaces	Convection and radiation
Bottom surface	Contact heat transfer to support table
Bottom contact conductance	3000 W/m²K (assumed)
Probe extraction method	Trilinear interpolation for interior locations
Backside temperature	Reconstructed using bottom-cell conduction and contact thermal resistance



2.5 Validation Results:
 
Figure 3. Comparison of peak temperatures predicted by the present Goldak double-ellipsoidal heat-source model (Present FVM method) with reference finite element (FEM) and infrared thermography (IR) results at different transverse distances from the weld centreline, including predictions at 6.75 mm depth and the reconstructed backside surface (7.5 mm).
2.6 Analysis Results:
 
Figure 3. Temperature contours at various time such as 0s, 9s, 18s, 27s and 36s.
 
Figure 4. Variation of temperature with welding time at different transverse distances from the weld centreline and through-thickness locations of the workpiece.
 
Figure 5. Variation of peak temperature across the plate thickness at different transverse distances from the weld centreline, illustrating the influence of depth and lateral position on the thermal response.
2.7 Energy Balance of the Thermal Simulation:
 Table 4. Global energy balance of the GMAW thermal simulation
Energy component	Value
Cumulative heat input	108.018 kJ
Boundary heat losses	74.433 kJ
Sensible energy stored	33.585 kJ
Energy balance residual	6.06 × 10⁻⁸ J
Relative residual	5.61 × 10⁻¹¹ %

3. Conclusions: 
(i) The Goldak double-ellipsoidal heat-source model successfully provides a mathematical representation of the spatially distributed and time-dependent heat input associated with the travelling GMAW arc. The independent front and rear ellipsoidal parameters enable the asymmetric distribution of thermal energy along the welding direction to be represented.
(ii) Three-dimensional transient thermal simulation: The developed explicit finite-volume solver captures the transient heat-conduction behaviour of the API 5L X80 steel plate under a moving volumetric heat source. Temperature-dependent thermal conductivity, specific heat capacity, and density are incorporated into the numerical formulation.
(iii) Temperature distribution across the workpiece: The predicted peak temperatures exhibit a decreasing trend with increasing transverse distance from the weld centreline. The temperature response also varies through the plate thickness, demonstrating the influence of both lateral position and depth on the thermal behaviour of the welded plate.
(iv) Comparison with reference results: The predicted peak temperatures at the 6.75 mm depth show closer agreement with the reference infrared thermography measurements, yielding a mean absolute error (MAE) of 33.95 °C and a root mean square error (RMSE) of 40.34 °C. At the reconstructed backside surface (7.5 mm), the corresponding MAE and RMSE are 76.80 °C and 86.53 °C, respectively. The backside predictions underestimate all eight reference infrared peak temperatures, indicating that further investigation of the thermal boundary conditions and through-thickness heat transfer is necessary.
(v) Numerical energy conservation: The global energy accounting demonstrates excellent numerical energy closure. The cumulative heat input of 108.018 kJ is distributed into 74.433 kJ of boundary heat losses and 33.585 kJ of sensible thermal energy stored in the workpiece. The final energy balance residual is approximately 6.06×times10-8 J, indicating that the implemented discrete energy balance is satisfied with very high numerical precision.
4. Physics-informed machine-learning workflow:
 
5. References:
1. Rocha, E. J. F., Antonino, T. de S., Guimarães, P. B., Ferreira, R. A. S., Barbosa, J. M. A., & Rohatgi, J. (2018). Modeling of the temperature field generated by the deposition of weld bead on a steel butt joint by FEM techniques and thermographic images. Materials Research, 21(3), e20160796. https://doi.org/10.1590/1980-5373-MR-2016-0796)


