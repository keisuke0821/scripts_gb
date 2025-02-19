# ============ import packages needed ============

import numpy as np
import glob
import sys
import pickle
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp

# ============ import gb package ============
sys.path.append('/home/keisuke/script/script/')
from gb_cal.misc.misc import get_moon_altaz_skm, _astro_pos, save_pkl, read_pkl
from gb_cal.readdata.planet_center import calc_rmat_mc, calc_mc_data
import mkid_pylibs as klib
from mkid_pylibs.rhea_comm.lib_read_rhea  import *
from analyzer_db.kidslist import KidsList

# ============ import gbirdproc. This is C++ package to treat kiddata. ============
### If you want to use logbdata in your own directory, please edit file_retriever.cxx in /src to your directory
sys.path.append('/home/keisuke/script/script/gbproc_cpp/build')
import gbirdproc as gbp

# ============ information about chip ID and array ID ============
CHIPS = np.array(['3A',   '2A',   '3B',   '1A',   '2B',   '1B',   '220'])
DAQS  = np.array(['GB01', 'GB02', 'GB03', 'GB04', 'GB05', 'GB06', 'GB07'])

# ============ directory to save files ============
SAVEDIR = '/data/keisuke/gb/save/'

class read_rawdata_cpp():
    def __init__(self, meas_id, debug = True, log = True, saveraw = True):
        self.meas_id = meas_id
        db = gbp.MeasDB(meas_id)
        self.daq = db.swppath.split('_')[-1][:4]
        self.chip = CHIPS[np.where(DAQS == self.daq)[0][0]]

        try:

            swp = gbp.RheaSwpReader(db.swppath,klist.lofreq)
            tod = gbp.RheaTodReader(db.todpath,klist.lofreq)
        except:
            from analyzer_db.kidslist import KidsList
            klist = KidsList(db.klistpath)
            swp = gbp.RheaSwpReader(db.swppath,klist.sg_freq)
            tod = gbp.RheaTodReader(db.todpath,klist.sg_freq)
            klist.blinds_freq = np.array(klist.blinds_freqs)
            klist.kids_freq = np.array(klist.kids_freqs)

        # ============ make swpset,todset ============
        swpset = [klib.Swpdata(ifreq,'I-Q',(iiq.real,iiq.imag)) for ifreq, iiq in zip(swp.freq, swp.iq)]
        todset = [[klib.TODdata(tod.time,'I-Q',(iiq.real, iiq.imag),ifreq, info={'n_rot':tod.syncnum,'sync_off':tod.syncoff})] for ifreq, iiq in zip(tod.freq, tod.iq)]

        # ============ check bline data quality ============
        good_bfreq = []
        for ibind, ibfreq in zip(klist.blinds_index, klist.blinds_freq):
            if np.abs(ibfreq) < 0.1:
                pass
            else:
                if any(np.abs(np.diff(todset[ibind][0].phase)) > 6):
                    print('============Blind TOD over 6. might be due to over 2pi==========')
                    ### consider the gap made by the jump in phase pi when reading out of kiddata
                    print(np.abs(np.diff(todset[ibind][0].phase))[np.abs(np.diff(todset[ibind][0].phase)) > 6])
                    pass
                else:
                    good_bfreq.append(ibfreq)
        good_bfreq = np.array(good_bfreq)

        # ============ set nearest bline tone ============
        binds = []
        for ifreq in klist.kids_freq:

            ind = np.where(np.min(np.abs(good_bfreq - ifreq)) == np.abs(klist.blinds_freq - ifreq))[0][0]
            binds.append(klist.blinds_index[ind])
        self.bind = binds

        kr = []
        for i, ibind in enumerate(binds):
            kr.append(klib.kidana_psd.PSDAnalyzer(swp=swpset[i], tod=todset[i], ctod=todset[ibind]))

        # ============ fitting ============
        nfwhm = -1
        usedindex = np.arange(0, int(len(kr)))
        nonphase = [None]*len(kr)
        rangeind, freqranges, twokidind, twokidfitter, initind, fitinit, depind, depval, skipind, guessskip = get_fitconfig(self.chip)
        for i, (ikr) in enumerate(kr):
            if i in initind:
                fitini = fitinit[np.where(np.array(initind) == i)[0][0]]
            else:
                fitini = None
            if i in rangeind:
                rangeini = freqranges[np.where(np.array(rangeind) == i)[0][0]]
            else:
                rangeini = [None,None]
            if i in twokidind:
                twokidini = twokidfitter[np.where(np.array(twokidind) == i)[0][0]]
            else:
                twokidini = 'gaolinbg'
            if i in depind:
                dep = depval[np.where(np.array(depind) == i)[0][0]]
            else:
                dep = 3
            if i in skipind:
                skip = guessskip[np.where(np.array(skipind) == i)[0][0]]
            else:
                skip = False
            print(f'======== Fit KID{i:02} ==========')
            ikr.fitIQ(nfwhm=-1, frqrange = rangeini, fitter = twokidini, init = fitini, dep = dep, guess_skip = skip)

            nonphase[i] = 2*np.tan(ikr.tod.rwmdata.corphase/2.) ### this is the phase response after the correction of nonlinearity

        # ============ read az el ============
        print("Reading AzEl data")
        ### get the syncronozed az and el data in logbdata
        az = gbp.get_syncaz_rhea(tod,41,False,False)
        el = gbp.get_syncel_rhea(tod,0,False,False)

        ### check az and el files exist in logbdata directory. If not, rsync manually and save in your directory

        # ============ read log. log data includes PWV, weather, and humidity etc... ============
        if log:
            print("Reading Log data")
            log = gbp.LogContainer(az.time)

            # ============ data selection ============
            goodinds = log.goodIndex(0.35,80)
            ### 0.35 -> detector temperature (K)
            ### 80 -> humidity (%)

            if len(goodinds) == 0:
                raise ValueError("goodinds is empty.")

            # ============ process with the number of goodindex ============
            ### usually, can get 2 goodindex and set the range between 2 goodindex as "True"

            if len(goodinds) == 1:
                ### set arbitrary goodindex
                goodinds = [100000,goodinds[0]]

            good = np.array([False]*(goodinds[-1]+1))

            if len(goodinds) > 1:
                isok = True
                for i,j in zip(goodinds[:-1],goodinds[1:]):
                    print(f"Setting good[{i}:{j+1}] to {isok}")
                    ### fill with isok in [i:j+1] of good array
                    good[i:j+1] = isok
                    ### inverse the value of isok (True <-> False)
                    isok = (not isok)

            # ============ remove glitch ============
            for i, (ikr) in enumerate(kr):
                thre = 1 ### threshold of glitch
                corphase_diff = np.ediff1d(ikr.tod.rwmdata.corphase, to_end=0)
                print(f"Detector {i}: corphase_diff = {corphase_diff}")
                glitch_ind = np.where(np.abs(corphase_diff) > thre)[0]
                print(f"Detector {i}: Number of glitches detected: {len(glitch_ind)}")
                print(f"Glitch indices for detector {i}: {glitch_ind}")
                glitch_ind = np.where(np.abs(np.ediff1d(ikr.tod.rwmdata.corphase, to_end = 0)) > thre)[0]
                for iglitch_ind in glitch_ind:
                    good[iglitch_ind] = False

            print("Final good array after glitch removal:" , good)

        # ============ raw data ============

        self.kr  = kr
        self.klist = klist
        self.phase = nonphase
        self.index = usedindex
        self.az    = az
        self.el    = el
        self.rpm   = az.speed
        self.time = np.array([datetime.fromtimestamp(itime, timezone.utc) for itime in az.time])

        if log:
            self.good  = good
            self.log   = log
        else:
            self.good = np.zeros(len(self.phase[0]), dtype=bool)
            self.good[1000:] = True

        if debug:
            self.cswp = swp
            self.ctod = tod
            self.kr  = kr
            self.param = get_param(self.kr)
            import os
            if not os.path.isdir(SAVEDIR + 'swp_param/'):
                os.mkdir(SAVEDIR + 'swp_param/')
            save_file = SAVEDIR + 'swp_param/' + self.chip + f'_{self.meas_id}' + '.csv'
            self.param.to_csv(save_file)
        if saveraw:
            self.save_rawdata_all()

    # ============ save rawdata, which consists of (utime, el, az , phase) ============

    def save_rawdata(self, kidid=0):
        ret = {}
        if isinstance(self.good, range) or isinstance(self.good, list):
            good_indices = list(self.good)
        else:
            good_indices = self.good
        if len(good_indices) == 0:
            raise ValueError("The 'good' array is empty.")

        ret['utime'] = self.az.time[good_indices]
        ret['el'] = self.el.angle[good_indices]
        ret['az'] = self.az.angle[good_indices]
        ret['phase'] = self.phase[kidid][good_indices]

        import pickle
        import os
        if not os.path.isdir(SAVEDIR + 'raw_data'):
            os.mkdir(SAVEDIR + 'raw_data')
        with open(SAVEDIR + 'raw_data/'  + '_{}'.format(self.meas_id) + '_kid{:02}.pkl'.format(kidid), mode="wb") as f:
            pickle.dump(ret, f)

    # ============ save rawdata of all kids in this meas_id =============
    def save_rawdata_all(self):
        for i in range(len(self.kr)):
            self.save_rawdata(i)

        print("save_rawdata_all finished!")

    # ============ basic plot ===========
        print("basic plot")

    def plot_log(self, save = True):
        print("start plot_log!")
        from gb_cal.readdata.plot import plot_log

        plot_log(self.time[self.good], self.log.thermo.sync_Tdet[self.good],
        self.log.gaulli.sync_pwv[self.good], self.log.bme.sync_humidity[self.good],
        save = save, meas_id = self.meas_id, daq = self.daq, chip = self.chip)

    def plot_swpamp(self, save = True):
        from gb_cal.readdata.plot import plot_swpamps
        plot_swpamps(self.kr, save = save, meas_id = self.meas_id, daq = self.daq, chip = self.chip)
    def plot_swpiq(self, save = True):
        from gb_cal.readdata.plot import plot_swpiqs
        plot_swpiqs(self.kr, self.good, save = save, meas_id = self.meas_id, daq = self.daq, chip = self.chip)
    def plot_psd(self, save = True):
        from gb_cal.readdata.plot import plot_psds
        for ikr in self.kr:
            ikr.calcpsd(dofit = True)
        plot_psds(self.kr, save = save, meas_id = self.meas_id, daq = self.daq, chip = self.chip)

    # ======= save swpparam ===========
    def save_swpparam(self):
        print("save_swpparam")
        params = get_param(self.kr, save = True, meas_id = self.meas_id, daq = self.daq, chip = self.chip)
        print(params)
        del params

    # ========== Moon centered data ===========

    # ============ get an index with which phase has max value, that index means the time observing the center of moon ============
    def calc_maxind(self):
        self.maxinds = []
        for iphase in self.phase:
            good_indices = self.good
            if len(good_indices) == 0:
                raise ValueError("The 'good' array is empty.")
            iphase_good = iphase[good_indices]
            if len(iphase_good) == 0:
                raise ValueError(f"The iphase array for good indices is empty: {iphase_good}")
            max_value = np.max(iphase_good)
            max_indices = np.where(iphase_good == max_value)[0]
            if len(max_indices) == 0:
                raise ValueError(f"No max value found in iphase_good: {iphase_good}")
            self.maxinds.append(max_indices[0])
        self.maxinds = np.array(self.maxinds)
        print("calc_maxinds is", self.maxinds)

    # ============ calculate offset value between moon position from astropy and telescope's angle information when observing the moon center ============
    def calc_offset(self, save = True):
        azoffs = []
        eloffs = []
        moon_az_list = []
        moon_el_list = []
        for i, imaxind in enumerate(self.maxinds):
            dttmp = self.time[self.good][imaxind] ### get the time corresponding to maxind
            moon = _astro_pos(dttmp, obj = 'moon') ### get the position of moon (az, el) in dttmp

            ### az and el of moon
            moon_az_list.append(moon.az.value)
            moon_el_list.append(moon.alt.value)

            ### calculate the offsets
            azoffs.append(moon.az.value  - (360 - (self.az.angle[self.good][imaxind])))
            eloffs.append(moon.alt.value - self.el.angle[self.good][imaxind])

        self.azoffs = azoffs
        self.eloffs = eloffs
        if save:
            self.save_offset()

    # ============ save the offsets ============
    def save_offset(self):
        off = {}
        off['chip'] = self.chip
        off['daq'] = self.daq
        off['meas_id'] = self.meas_id
        off['kidid'] = np.arange(0, len(self.kr))
        off['azoff'] = self.azoffs
        off['eloff'] = self.eloffs
        save_pkl(off, f'/data/keisuke/gb/save/offset_tmp/offset_{self.meas_id}_' + self.daq + '_' + self.chip + '_noazoff.pkl')

    # ============ calculate moon centered data ============
    def calc_mc(self):
        ### matrix to rotate the angles to moon centered coordinates
        rmat_inv = calc_rmat_mc(self.az.time[self.good])
        self.mc_theta, self.mc_phi = calc_mc_data(self.az.angle[self.good], self.el.angle[self.good], self.azoffs, self.eloffs, rmat_inv)

    # ============ make moon centered data ============
    def make_mcdata(self, kidid=0):
        ret = {}
        ret['kidid'] = kidid ### kidid
        ret['utime'] = self.az.time[self.good] ### utime
        ret['phase'] = self.phase[kidid][self.good]  ### phase data
        ret['theta'] = np.array(self.mc_theta[kidid]) ### el
        ret['phi'] = np.array(self.mc_phi[kidid]) ### az
        return ret

    # ============ save the moon centered data ============
    def save_mcdata(self, kidid=0, ran = 2.0):
        fitdata = get_limdatasq(self.make_mcdata(kidid), ran = ran)
        save_pkl(fitdata, f'/data/keisuke/gb/save/mcdata/mcdata_{self.meas_id}_' + self.daq + '_' + self.chip + f'_kid{kidid:02}' + '.pkl')
        del fitdata

    # ============ save the moon centered data of all kids ============
    def save_mcdata_all(self):
        for ikidid in range(len(self.phase)):
            self.save_mcdata(ikidid)

    # ============ summary script for moon centered data ============
    def mc_data_all(self, save = True):
        self.calc_maxind()
        self.calc_offset(save = save)
        self.calc_mc()
        if save:
            self.save_mcdata_all()

    # =========== mcmap in healpix ==========
    def calc_mc_map(self, ckid , nside = 1024):
        self.ckid = ckid
        hitmaps = []
        hpxmaps = []
        maxpixs = []
        npix = hp.nside2npix(nside)
        sum_hpxmap = np.zeros(npix)
        for ikid in range(len(self.phase)):
            npix = hp.nside2npix(nside)
            ipix = hp.ang2pix(nside, theta=np.deg2rad(self.mc_theta[ckid]), phi=np.deg2rad(self.mc_phi[ckid]))
            maxpix =  hp.ang2pix(nside, theta=np.deg2rad(self.mc_theta[ckid][self.maxinds[ikid]]), phi=np.deg2rad(self.mc_phi[ckid][self.maxinds[ikid]]))
            hitmap = np.full(npix,hp.pixelfunc.UNSEEN)
            ind,n = np.unique(ipix, return_counts=True)
            hitmap[ind] = n
            hpxmap = np.zeros(npix)
            np.add.at(hpxmap, ipix, self.phase[ikid][self.good]/np.max(self.phase[ikid][self.good]))
            hpxmap[hitmap==hp.UNSEEN] = hp.UNSEEN
            hpxmap[hitmap!=hp.UNSEEN] /= hitmap[hitmap!=hp.UNSEEN]
            hitmaps.append(hitmap)
            hpxmaps.append(hpxmap)
            maxpixs.append(maxpix)
        for ikid in range(len(self.phase)):
            sum_hpxmap[hitmaps[ikid]!=hp.UNSEEN] += hpxmaps[ikid][hitmaps[ikid]!=hp.UNSEEN]

        self.hitmaps = hitmaps
        self.hpxmaps = hpxmaps
        self.maxpixs = maxpixs
        self.sum_hpxmap = sum_hpxmap
        del hitmaps, hpxmaps, maxpixs, sum_hpxmap

    def plot_mceach(self, save = True):
        plt.figure(figsize = (30,30))
        for i in range(len(self.phase)):
            plt.subplot(5,5,i+1)
            hp.gnomview(self.hpxmaps[i],hold=True,reso = 2, rot = (0,0,0), title = 'kid{:02}'.format(i))
            hp.graticule()
        plt.suptitle(f'mceach_{self.meas_id}_' + self.daq + '_' + self.chip + '.jpg')
        plt.tight_layout()
        if save:
            plt.savefig(f'/data/keisuke/gb/save/plot/mceach/mceach_center_kid{self.ckid:02}_{self.meas_id}_' + self.daq + '_' + self.chip + '.jpg')

    def plot_mcall(self, save = True):
        plt.figure(figsize = (8,8))
        hp.gnomview(self.sum_hpxmap, hold=True, reso = 2, rot = (0,0,0), title = f'mcall_{self.meas_id}_' + self.daq + '_' + self.chip)
        hp.graticule()
        for ind in range(len(self.maxinds)):
            hp.projtext(np.deg2rad(self.mc_theta[self.ckid][self.maxinds[ind]]), np.deg2rad(self.mc_phi[self.ckid][self.maxinds[ind]]), '{:02}'.format(ind), lonlat=False, color='r',size=10)
        if save:
            plt.savefig(f'/data/keisuke/gb/save/plot/mcall/mcall_center_kid{self.ckid:02}_{self.meas_id}_' + self.daq + '_' + self.chip + '.jpg')

    # ========= summary script for mc map =======
    def mc_map_all(self, ckid, save = True):
        self.calc_mc_map(ckid = ckid)
        self.plot_mceach(save = save)
        self.plot_mcall(save = save)

def get_param(kr, save = True, meas_id = 0000, daq = 'GB00', chip = 'test'):
    df = pd.DataFrame()
    fr = []
    Qr = []
    Qc = []
    Qi = []
    params = [fr, Qr, Qc, Qi]
    params_str = ['fr', 'Qr', 'Qc', 'Qi']
    for idata in kr:
        if 'fr' in idata.swp.fitresult.fitparamdict:
            for iparam, istr in zip(params, params_str):
                iparam.append(idata.swp.fitresult.fitparamdict[istr])
        else:
            if 'Qi1' in idata.swp.fitresult.fitparamdict:
                for iparam, istr in zip(params, params_str):
                    iparam.append(idata.swp.fitresult.fitparamdict[istr + '1'])
            else:
                for iparam, istr in zip(params, params_str):
                    iparam.append(idata.swp.fitresult.fitparamdict[istr + '2'])

    for iparam, istr in zip(params, params_str):
        df[istr] = iparam
    if save:
        df.to_csv(f'/data/keisuke/gb/save/swp_param/swpparam_{meas_id}_' + daq + '_' + chip + '.csv')
    return df

def get_limdata(mc_data, r = 1.5):
    tmp_theta = mc_data['theta'] - 90
    tmp_phi = (mc_data['phi'] - 180)%360 - 180
    moon_cond = (tmp_theta**2 + tmp_phi**2) < r**2
    fitdata = {}
    fitdata['utime'] = mc_data['utime'][moon_cond]
    fitdata['theta'] = mc_data['theta'][moon_cond] - 90
    fitdata['phi'] = (mc_data['phi'][moon_cond] - 180)%360 - 180
    fitdata['phase'] = mc_data['phase'][moon_cond]
    return fitdata

def get_limdatasq(mc_data, ran = 1.5):
    tmp_theta = mc_data['theta'] - 90
    tmp_phi = (mc_data['phi'] - 180)%360 - 180
    moon_cond = (np.abs(tmp_theta) < ran) & (np.abs(tmp_phi) < ran)
    fitdata = {}
    fitdata['utime'] = mc_data['utime'][moon_cond]
    fitdata['theta'] = mc_data['theta'][moon_cond] - 90
    fitdata['phi'] = (mc_data['phi'][moon_cond] - 180)%360 - 180
    fitdata['phase'] = mc_data['phase'][moon_cond]
    return fitdata

def destripe(mc_subdata, r = 1.5, maskr = 1):
    phase_div = []
    phase_mean = []
    phase_res = []
    phi_div = []
    utime_div = []
    theta_div = []
    phi_div = []
    period_ind = np.where((np.diff(mc_subdata['utime']) > 1) == True)[0]
    for i, iperiod_ind in enumerate(period_ind[:-1]):
        phase_div.append(mc_subdata['phase'][period_ind[i]:period_ind[i+1]])
        phi_div.append(mc_subdata['phi'][period_ind[i]:period_ind[i+1]])
        theta_div.append(mc_subdata['theta'][period_ind[i]:period_ind[i+1]])
        utime_div.append(mc_subdata['utime'][period_ind[i]:period_ind[i+1]])
        tmpmask = ~((mc_subdata['theta'][period_ind[i]:period_ind[i+1]]**2 + mc_subdata['phi'][period_ind[i]:period_ind[i+1]]**2) < maskr**2)
        mean = np.mean(mc_subdata['phase'][period_ind[i]:period_ind[i+1]][tmpmask])
        phase_mean.append(mean)
        phase_res.append(mc_subdata['phase'][period_ind[i]:period_ind[i+1]] - mean)

    phaseres = np.array([x for row in phase_res for x in row])
    phaseraw = np.array([x for row in phase_div for x in row])
    phires = np.array([x for row in phi_div for x in row])
    thetares = np.array([x for row in theta_div for x in row])
    utimeres = np.array([x for row in utime_div for x in row])
    ret = {}
    ret['phase'] = phaseres
    ret['phaseraw'] = phaseraw
    ret['phi'] = phires
    ret['theta'] = thetares
    ret['utime'] = utimeres
    return ret

def get_fitconfig(chip):
    if True:
        if chip == '1A':
            rangeind = [15]
            freqranges = [(4.886e9, 4.8885e9)]
            twokidind = [5, 6]
            twokidfitter = ['gaolinbg2f', 'gaolinbg2l']
            initind = []
            fitinit = []
            depind = []
            dep = []
            skipind = []
            guessskip = []
            return rangeind, freqranges, twokidind, twokidfitter, initind, fitinit, depind, dep, skipind, guessskip
        elif chip == '1B':
            rangeind = [10, 17, 19, 20, 21, 22] # KID2,3,4 are degenerate.
            freqranges = [(4.8105e9, 4.8125e9), (4.878e9, 4.881e9), (4.888e9, 4.890e9), (4.890e9, 4.892e9), (4.894e9, 4.8964e9), (4.8965e9, 4.899e9)]
            twokidind = [9, 17, 18]
            twokidfitter = ['gaolinbg2f', 'gaolinbg2f', 'gaolinbg2l']
            initind = []
            fitinit = []
            depind = []
            dep = []
            skipind = []
            guessskip = []
            return rangeind, freqranges, twokidind, twokidfitter, initind, fitinit, depind, dep, skipind, guessskip
        elif chip == '2A':
            rangeind = []
            freqranges = []
            twokidind = []
            twokidfitter = []
            initind = []
            fitinit = []
            depind = []
            dep = []
            skipind = []
            guessskip = []
            return rangeind, freqranges, twokidind, twokidfitter, initind, fitinit, depind, dep, skipind, guessskip
        elif chip == '2B':
            rangeind = []
            freqranges = []
            twokidind = []
            twokidfitter = []
            initind = []
            fitinit = []
            depind = []
            dep = []
            skipind = []
            guessskip = []
            return rangeind, freqranges, twokidind, twokidfitter, initind, fitinit, depind, dep, skipind, guessskip
        elif chip == '3A':
            rangeind = [7, 8, 9, 10, 13, 14]
            freqranges = [(4.736e9, 4.7385e9), (4.736e9, 4.7385e9), (4.7385e9, 4.741e9), (4.7385e9, 4.741e9), (4.7825e9, 4.785e9), (4.7825e9, 4.785e9)]
            #twokidind = [7, 8, 9, 10, 13, 14]
            #twokidfitter = ['gaolinbg2f', 'gaolinbg2l', 'gaolinbg2f', 'gaolinbg2l', 'gaolinbg2f', 'gaolinbg2l']
            twokidind = [7, 8, 10, 13, 14]
            twokidfitter = ['gaolinbg2f', 'gaolinbg2l', 'gaolinbg2l', 'gaolinbg2f', 'gaolinbg2l']
            #initind = [7, 8, 9, 10, 13, 14]
            initind = [7, 8, 10, 13, 14]
            fitinit = [{'fr1' : 4.7373e9, 'fr2' : 4.7376e9, 'Qr1': 15000, 'Qr2': 30000, 'Qc1': 15000, 'Qc2': 30000, 'arga': 0, 'absa': 0.01, 'tau': 0, 'phi01': 0, 'phi02': 0, 'c': 0 }, # 7
                       {'fr1' : 4.7373e9, 'fr2' : 4.7376e9, 'Qr1': 15000, 'Qr2': 30000, 'Qc1': 15000, 'Qc2': 30000, 'arga': 0, 'absa': 0.01, 'tau': 0, 'phi01': 0, 'phi02': 0, 'c': 0 }, # 8
                       #{'fr1' : 4.7394e9, 'fr2' : 4.7399e9, 'Qr1': 10000, 'Qr2': 30000, 'Qc1': 17000, 'Qc2': 40000, 'arga': 0, 'absa': 0.01, 'tau': 0, 'phi01': 0, 'phi02': 0, 'c': 0 }, # 9
                       {'fr1' : 4.7395e9, 'fr2' : 4.7399e9,'Qr1': 16000, 'Qr2': 40000, 'Qc1': 17000, 'Qc2': 40000}, # 10
                       {'fr1' : 4.7836e9, 'fr2':4.7842e9, 'Qr1': 15000, 'Qr2': 35000, 'Qc1': 10000, 'Qc2': 20000}, # 13
                       {'fr1' : 4.7836e9, 'fr2':4.7842e9, 'Qr1': 15000, 'Qr2': 35000, 'Qc1': 10000, 'Qc2': 20000}] # 14
            depind = [8, 9]
            dep = [2, 2]
            #skipind = [7, 8, 9]
            #guessskip = [True, True, True]
            skipind = [7]
            guessskip = [True]

            return rangeind, freqranges, twokidind, twokidfitter, initind, fitinit, depind, dep, skipind, guessskip
        elif chip == '3B':
            #rangeind = [0, 12]
            #freqranges = [(4.6858e9, 4.6868e9), (4.819e9, 4.821e9)]
            rangeind = [0]
            freqranges = [(4.68575e9, 4.6868e9)]

            twokidind = []
            twokidfitter = []
            initind = [0]
            fitinit = [{'fr' : 4.6862e9, 'Qr' : 15000, 'Qc' : 6000, 'phi0' : 0.1}]
            depind = []
            dep = []
            skipind = []
            guessskip = []
            return rangeind, freqranges, twokidind, twokidfitter, initind, fitinit, depind, dep, skipind, guessskip
        elif chip == '220':
            rangeind = [0, 2, 15]
            freqranges = [(4.854e9, 4.857e9), (4.867e9, 4.869e9), (4.993e9, 4.995e9)]
            twokidind = [14]
            twokidfitter = ['gaolinbg2f']
            initind = []
            fitinit = []
            depind = []
            dep = []
            skipind = []
            guessskip = []
            return rangeind, freqranges, twokidind, twokidfitter, initind, fitinit, depind, dep, skipind, guessskip
        else:
            print('Error : chip:' + chip + 'is not matched.')
