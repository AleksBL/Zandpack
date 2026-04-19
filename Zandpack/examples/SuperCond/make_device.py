from Zandpack.TimedependentTransport import TD_Transport
import sisl
import matplotlib.pyplot as plt
import numpy as np

t_dev =  -2.7
t_elec = -2.7
lat_const = 1.0
line = np.linspace(-9.0,9.0,151) + 1j*1e-2
line = np.vstack((line,line))

c = sisl.geom.sc(lat_const,'H').tile(2,1).add_vacuum(10,1).add_vacuum(10,2)
c.set_nsc((3,1,1))

for i in range(c.no):
    c.atoms[i] = sisl.Atom('H', R = 2.0)

geom_dev = c.tile(5,0)
geom_dev.xyz[4,1] -= 0.2
geom_dev.xyz[5,1] += 0.2
geom_dev = geom_dev.add(sisl.Geometry(geom_dev.xyz[4]-np.array([0,1,0]) )).add(sisl.Geometry(geom_dev.xyz[5]+np.array([0,1,0])))
geom_em  = c.copy()
geom_ep  = c.copy().move(geom_dev.cell[0,:] - geom_em.cell[0,:])
#geom_dev = geom_dev.remove([10])

sisl.plot(geom_dev); plt.axis('equal')
plt.show()

Test = TD_Transport([geom_em,geom_ep], geom_dev, kT_i = [0.01, 0.01])
Test.Make_Contour(line, 10, pole_mode = 'JieHu2011', save_pics=True)

Test.Electrodes( semi_infs = ['-a1', '+a1'] )
Test.make_device(elec_inds = [[i for i in range(2)],[i + 2*4 for i in range(2)]])

elec = sisl.Hamiltonian(c)
elec.construct([[0.1, lat_const * 1.1], 
                [0  , t_elec         ] ])

Test.run_electrodes(fois_gras_H = [elec, elec])
dev_H = sisl.Hamiltonian(geom_dev)
dev_H.construct([[0.1, lat_const * 1.1], 
                 [0,   t_dev             ] ])
fact = 2.2
dev_H[11,5] = fact*t_dev
dev_H[5,11] = fact*t_dev
dev_H[10,4] = fact*t_dev
dev_H[4,10] = fact*t_dev
dev_H[10,10] = 0.66*t_dev
dev_H[11,11] = 0.66*t_dev

Test.run_device(fois_gras_H = dev_H)
Test.read_data()
Test.pickle('Test')
