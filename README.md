# Bragg-Magnifier-RT

This is a Ray Tracing Simulation for a system of optically polished monocrystal (possibly asymmetrically cut) erradiated by a narrow polychromatic X- ray beam of an extended  X-ray source. 
A ray tracing cycle can be divided into two stages: a) setting the position of a central axis (CA) - single ray with wavelength Lambda_0 through which the spatial positions and orientations 
of optical components are defined, and b) tracing of rays, where the light beam propagation through the system is simulated. 
Light source is simulated by flat rectangular matrix of point sources geometrical dimension of which are set accordingly to synchrotron specifications.
Every ray originates in one of these point sources and its initial direction is set randomly within the values of opening angle of the source.	After every reflection from a 
crystal surface the amplitude of a ray is scaled down on a factor which is defined by the angle of its incidence.
The angles between all reflective surfaces and the CA are ‘tuned’ to be in the center of diffraction region for the light with wavelength Lambda_0, 
however for the rays with  wavelength, Lambda_0+ Del_lambda,  which even propagate precisely along the CA, diffraction condition may not be fulfilled.. See function Crystals.py\core_angle.py used in \crystal_reflection.py
Files of the program: 'RayTracer_LabSys_4crystals.py' has the full simulation cycle that runs the number of iterations Parameters.txt->"repeat". Result is the array histograms_for_comparison which can be plot together with experimental results in file 'run.py' 'Ray visualizer.py' contians the main classes, 'Crystal.py' - dynamical diffraction mathematics, 'Ray_plane_interaction' - the math of the ray tracing. 
### Running the program
Run the file 'run.py'. 
All the parameters of the beamline are to be stores in files with names: 'Parameters_{beamline_name}.txt' in the form of a js-dictionary. Variables: all_crystals, Rocking_pos_mono, and distances contain the parameters of the system geometry. In the list 'crystals_normals' one can also specify the mistune-scan of any existing crystal - instead of the monochromator_crystal_2.]
