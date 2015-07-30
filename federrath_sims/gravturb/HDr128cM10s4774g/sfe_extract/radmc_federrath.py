"""
1. Need to write temperature, density, velocity, grid
2. Need to copy over dust opacity
3. Compute the line populations first: 
   (need to set lines_mode = 3 before doing this)
    radmc3d calcpop
4. Then compute the "image":
    radmc3d image iline 1 widthkms 10 linenlam 40 linelist nostar


"""
from yt.analysis_modules.radmc3d_export.api import RadMC3DWriter
from yt.utilities.physical_constants import kboltz
import yt

ds = yt.load('GT_hdf5_plt_cnt_0204')

x_co = 1.0e-4
x_h2co = 1.0e-9
mu_h = yt.YTQuantity(2.34e-24, 'g')
def _NumberDensityCO(field, data):
    return (x_co/mu_h)*data["density"]
yt.add_field(("gas", "number_density_CO"), function=_NumberDensityCO, units="cm**-3")
def _NumberDensityH2(field, data):
    return (1./mu_h)*data["density"]
yt.add_field(("gas", "number_density_H2"), function=_NumberDensityH2, units="cm**-3")
def _NumberDensityH2CO(field, data):
    return (1./mu_h)*data["density"]*x_h2co
yt.add_field(("gas", "number_density_H2CO"), function=_NumberDensityH2CO, units="cm**-3")
dust_to_gas = 0.01
def _DustDensity(field, data):
    return dust_to_gas * data["density"]
ds.add_field(("gas", "dust_density"), function=_DustDensity, units="g/cm**3")

def _GasTemperature(field, data):
    # P V = rho k T
    # T = P V / (rho K)
    #return (data['pressure'].in_cgs().value/data['number_density_H2'].in_cgs().value/kboltz.in_cgs().value)*yt.units.K
    return (data['pressure'].value*0 + 20) * yt.units.K
ds.add_field(("gas", "temperature"), function=_GasTemperature, units="K")

writer = RadMC3DWriter(ds)
writer.write_dust_file(("gas", "temperature"), "gas_temperature.inp")

writer.write_amr_grid()
writer.write_line_file(("gas", "number_density_CO"), "numberdens_co.inp")
writer.write_line_file(("gas", "number_density_H2"), "numberdens_h2.inp")
writer.write_line_file(("gas", "number_density_H2CO"), "numberdens_h2co.inp")
writer.write_dust_file(("gas", "dust_density"), "dust_density.inp")
velocity_fields = ["velocity_x", "velocity_y", "velocity_z"]
writer.write_line_file(velocity_fields, "gas_velocity.inp")
# need to dump gas or dust temperature
# also dump dust density separately (dust_density.inp)


import radmc3dPy
import os
import subprocess
import shutil

# don't do these, they are for 1d examples and create crap files
#problem = 'lines_nlte_lvg_1d_1'
#radmc3dPy.analyze.writeDefaultParfile(problem)
#radmc3dPy.setup.problemSetupDust(problem)
#radmc3dPy.setup.problemSetupGas(problem)
shutil.copy('/Users/adam/repos/radmc-3d/version_0.39/python/python_examples/datafiles/dustkappa_silicate.inp', '.')
shutil.copy('/Users/adam/repos/radmc-3d/version_0.39/python/python_examples/datafiles/molecule_co.inp', '.')
shutil.copy('/Users/adam/LAMDA/ph2co-h2.dat','molecule_h2co.inp')

params=dict(istar_sphere=0, itempdecoup=1, lines_mode=3, nphot=1000000,
            nphot_scat=30000, nphot_spec=100000, rto_style=3,
            scattering_mode=0, scattering_mode_max=1, tgas_eq_tdust=1,)

params_string = """
istar_sphere = {istar_sphere}
itempdecoup = {itempdecoup}
lines_mode = {lines_mode}
nphot = {nphot}
nphot_scat = {nphot_scat}
nphot_spec = {nphot_spec}
rto_style = {rto_style}
scattering_mode = {scattering_mode}
scattering_mode_max = {scattering_mode_max}
tgas_eq_tdust = {tgas_eq_tdust}
"""

with open('radmc3d.inp','w') as f:
    params['lines_mode'] = 50
    f.write(params_string.format(**params))

assert os.system('radmc3d calcpop writepop') == 0

with open('radmc3d.inp','w') as f:
    params['lines_mode'] = 3
    f.write(params_string.format(**params))

# compute the dust temperature
assert os.system('radmc3d mctherm') == 0

# iline: 1 = CO 1-0, 2 = CO 2-1, etc.
# widthkms = full width of output spectrum, divided by linenlam
# linenlam: number of wavelengtyh bins
# linelist
radmc3dPy.image.makeImage(iline=1, imolspec=2, widthkms=5, linenlam=40, nostar=True)
shutil.move('image.out', 'h2co_1-0_image.out')
radmc3dPy.image.makeImage(iline=3, imolspec=2, widthkms=5, linenlam=40, nostar=True)
shutil.move('image.out', 'h2co_303-202_image.out')
radmc3dPy.image.makeImage(iline=13, imolspec=2, widthkms=5, linenlam=40, nostar=True)
shutil.move('image.out', 'h2co_321-220_image.out')
radmc3dPy.image.makeImage(iline=2, widthkms=5, linenlam=40, nostar=True)
shutil.move('image.out', 'co_2-1_image.out')
radmc3dPy.image.makeImage(iline=1, widthkms=5, linenlam=40, nostar=True)
shutil.move('image.out', 'co_1-0_image.out')
#radmc3d image iline 1 widthkms 10 linenlam 40 linelist nostar

# this didn't work probably b/c the size is wrong
#radmc3dPy.image.makeImage(npix=400, sizeau=10*206265., wav=3000., incl=0, iline=1, widthkms=10, linenlam=500,)

#imag=radmc3dPy.image.readImage()
#radmc3dPy.image.plotImage(imag, arcsec=True, dpc=14., log=True, maxlog=5)
#'radmc3d image npix '+str(DIMS[0])+' iline 1 widthkms 10 linenlam 500 loadlambda fluxcons inclline linelist nostar writepop doppcatch sizepc 10'
