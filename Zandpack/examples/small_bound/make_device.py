from TimedependentTransport.TimedependentTransport import TD_Transport
import sisl
import matplotlib.pyplot as plt
import numpy as np

t_dev  = 1
t_elec = 1
pad    = 5
lat_const = 1.0
line = np.linspace(-3.0,3.0,601) + 1j*1e-2
line = np.vstack((line,line))

c = sisl.geom.sc(1.0, sisl.Atom('C', R = 1.2)).add(sisl.geom.sc(1.0, sisl.Atom('C', R = 1.2)).move([0.5, 0, 0]))
c = c.tile(1,1).add_vacuum(10,1).add_vacuum(10,2)

geom_dev = c.tile(5 + 2 * pad,0)
geom_em = c.copy()
geom_ep = c.copy().move(geom_dev.cell[0,:] - geom_em.cell[0,:])

Test = TD_Transport([geom_em,geom_ep], geom_dev, kT_i = [0.025, 0.025], mu_i = [-.5, -.5])
Test.Make_Contour(line, 15, pole_mode = 'JieHu2011')

Test.Electrodes( semi_infs = ['-a1', '+a1'] )
Test.make_device(elec_inds = [[i for i in range(2)],[i + 2*(4+2*pad) for i in range(2)]]
                 )

elec1 = sisl.Hamiltonian(c)
elec1.set_nsc((3,1,1))
elec1.construct([[0.1, 0.6, 1.1], [0.0, t_elec, 0.0]])
elec1[0,0] = -0.5
elec1[1,1] = +0.5
elec2 = sisl.Hamiltonian(c)
elec2.set_nsc((3,1,1))
elec2.construct([[0.1, 0.6, 1.1], [0.0, t_elec, 0.0]])
elec2[0,0] = +0.5
elec2[1,1] = -0.5


geom_dev = geom_dev.add_vacuum(5,0)
dev_H = sisl.Hamiltonian(geom_dev)
dev_H.set_nsc((1,1,1))
dev_H.construct([[0.1, 0.6], [0.0, t_dev]])
for i in range(pad-1):
    dev_H[2*i+2,2*i+2] = -0.5
    dev_H[2*i+3,2*i+3] = +0.5
for i in range(pad-2):
    dev_H[dev_H.no-3-2*i,dev_H.no-3-2*i] = -0.5
    dev_H[dev_H.no-4-2*i,dev_H.no-4-2*i] = +0.5
    
dev_H[3+pad,3+pad] = 0.5
dev_H[dev_H.no - 4 - pad,
      dev_H.no - 4 - pad] = 0.5

dev_H[dev_H.no - 5 - pad,
      dev_H.no - 5 - pad] = 0.5

Test.run_electrodes(fois_gras_H = [elec1, elec2])
Test.run_device(fois_gras_H = dev_H)
Test.read_data()
tbt = sisl.get_sile('Device/siesta.TBT.nc')
Test.pickle('Test')
plt.show()

plt.plot(tbt.E[Test.sampling_idx[0]], 
         tbt.DOS()[Test.sampling_idx[0]]
         - tbt.ADOS(elec = 0)[Test.sampling_idx[0]]
         - tbt.ADOS(elec = 1)[Test.sampling_idx[0]]
         )
plt.title('Bound DOS')
plt.show()

plt.plot(tbt.E[Test.sampling_idx[0]], tbt.transmission()[Test.sampling_idx[0]])
plt.title('Transmission')
plt.show()
