import cantilevers
import materials
import laminate_fem
import analysers

nelx = 47 # stiffness 40 N/m
nelx = 70 # stiffness 60 N/m
nelx = 93 # stiffness 80 N/m
nelx = 115 # stiffness 100 N/m
nely = 50

material = materials.PiezoMumpsMaterial()
cantilever = cantilevers.RectangularCantilever(1e-6, 5e-6, nelx, nely)
cantilever = cantilevers.StandardB()
#cantilever = cantilevers.StandardD()
fem = laminate_fem.LaminateFEM(cantilever, material)
analyser = analysers.CantileverAnalyser(fem)

#analyser.plot_densities('plots/ref-stdb.png')
analyser.identify_modal_parameters(3)

