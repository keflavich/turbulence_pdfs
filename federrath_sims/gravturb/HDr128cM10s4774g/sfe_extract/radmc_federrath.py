from yt.analysis_modules.radmc3d_export.api import RadMC3DWriter
import yt

ds = yt.load('GT_hdf5_plt_cnt_0204')

x_co = 1.0e-4
mu_h = yt.YTQuantity(2.34e-24, 'g')
def _NumberDensityCO(field, data):
    return (x_co/mu_h)*data["density"]
yt.add_field(("gas", "number_density_CO"), function=_NumberDensityCO, units="cm**-3")
def _NumberDensityH2(field, data):
    return (1./mu_h)*data["density"]
yt.add_field(("gas", "number_density_H2"), function=_NumberDensityCO, units="cm**-3")
dust_to_gas = 0.01
def _DustDensity(field, data):
    return dust_to_gas * data["density"]
ds.add_field(("gas", "dust_density"), function=_DustDensity, units="g/cm**3")

def _DustTemperature(field, data):
    return data['pres']/data['density']
ds.add_field(("gas", "dust_temperature"), function=_DustTemperature, units="K")

writer = RadMC3DWriter(ds)
writer.write_dust_file(("gas", "dust_temperature"), "dust_temperature.inp")

writer.write_amr_grid()
writer.write_line_file(("gas", "number_density_CO"), "numberdens_co.inp")
writer.write_line_file(("gas", "number_density_H2"), "numberdens_h2.inp")
writer.write_dust_file(("gas", "dust_density"), "dust_density.inp")
velocity_fields = ["velocity_x", "velocity_y", "velocity_z"]
writer.write_line_file(velocity_fields, "gas_velocity.inp")
# need to dump gas or dust temperature
# also dump dust density separately (dust_density.inp)


import radmc3dPy
import os
import shutil

problem = 'lines_nlte_lvg_1d_1'
radmc3dPy.analyze.writeDefaultParfile(problem)
radmc3dPy.setup.problemSetupDust(problem)
radmc3dPy.setup.problemSetupGas(problem)
shutil.copy('/Users/adam/repos/radmc-3d/version_0.39/python/python_examples/datafiles/dustkappa_silicate.inp', '.')
shutil.copy('/Users/adam/repos/radmc-3d/version_0.39/python/python_examples/datafiles/molecule_co.inp', '.')

os.system('radmc3d mctherm')
#radmc3dPy.image.makeImage(npix=400, sizeau=10*206265., wav=3000., incl=0, iline=1, widthkms=10, linenlam=500,)

#imag=radmc3dPy.image.readImage()
#radmc3dPy.image.plotImage(imag, arcsec=True, dpc=14., log=True, maxlog=5)
#'radmc3d image npix '+str(DIMS[0])+' iline 1 widthkms 10 linenlam 500 loadlambda fluxcons inclline linelist nostar writepop doppcatch sizepc 10'
