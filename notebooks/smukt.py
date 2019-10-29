import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.stats import LombScargle, sigma_clip
from scipy.ndimage import gaussian_filter1d as gaussfilt


from matplotlib import rc
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplots, subplot, setp
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from seaborn import despine



mpl.style.use('seaborn-colorblind')

#To make sure we have always the same matplotlib settings
#(the ones in comments are the ipython notebook settings)

mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=18               #10 
mpl.rcParams['savefig.dpi']= 200             #72 
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

colours = mpl.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams["font.family"] = "Times New Roman"

import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

###-----------------------------------------------------------------
###
###-----------------------------------------------------------------


def get_pgram(time,flux, min_p=1./24., max_p = 20.):
    finite = np.isfinite(flux)
    ls = LombScargle(time[finite],flux[finite],normalization='psd')
    
    frequency, power = ls.autopower(minimum_frequency=1./max_p,maximum_frequency=1./min_p,samples_per_peak=5)

    norm = np.nanstd(flux * 1e6)**2 / np.sum(power) # normalize according to Parseval's theorem - same form used by Oliver Hall in lightkurve
    fs = np.mean(np.diff(frequency*11.57)) # ppm^/muHz

    power *= norm/fs

    spower = gaussfilt(power,15)

    return frequency, power, spower 


def plot_lc(ax1,time,lc,name,trends=None,title=False):
        m = (lc>0.) * (np.isfinite(lc))

        ax1.plot(time[m],lc[m]/np.nanmedian(lc[m]),'.')
        dt = np.nanmedian(time[m][1:]-time[m][:-1])
        ax1.set_xlim(time[m].min()-dt,time[m].max()+dt)
        if trends is not None:
            for j, trend in enumerate(trends):
                ax1.plot(time[m],trend[m]/np.nanmedian(trend[m]),'-',color=colours[2-j])
                # plt.legend(labels=['Flux','Trend'])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Relative Flux')
        if title:
            plt.title(r'%s' % name)

def plot_pgram(ax1,frequency,power,spower,name,min_p=1./24.,max_p=20.,title=False,fontsize=14,alias=1):

    ax1.plot(frequency*11.57,power,'0.8',label='Raw')
    ax1.plot(frequency*11.57,spower,linewidth=3.0,label='Smoothed')
    ax1.set_xlim(1./max_p*11.57,1./min_p*11.57)
    ax1.set_xlabel(r'Frequency ($\mu$Hz)')
    ax1.set_ylabel(r'Power (ppm$^2/{\mu}Hz)$')

    ax2 = ax1.twiny()
    ax2.tick_params(axis="x",direction="in", pad=-20)

    ax2.set_xlim(1./max_p,1./min_p)
    ax2.set_xlabel('c/d',labelpad=-20)
    plt.ylim(0,np.max(power));
    if title:
        plt.title(r'%s Power Spectrum' % name,y=1.01)
    ax1.legend()
    ax1.text(frequency[np.argmax(power)]*11.57*1.2,np.nanmax(power)*0.7,'P = %.3f d' % (1./frequency[np.argmax(power)]*alias),fontsize=fontsize)

def plot_log_pgram(ax1,frequency,power,spower,name,min_p=1./24.,max_p=20.,title=False):

    ax1.plot(frequency*11.57,power,'0.8',label='Raw')
    ax1.plot(frequency*11.57,spower,linewidth=3.0,label='Smoothed')
    ax1.set_xlim(1./max_p*11.57,1./min_p*11.57)
    ax1.set_xlabel(r'Frequency ($\mu$Hz)')
    ax1.set_ylabel(r'Power (ppm$^2/{\mu}Hz)$')

    ax2 = ax1.twiny()
    ax2.tick_params(axis="x",direction="in", pad=-20)

    ax2.set_xlim(1./max_p,1./min_p)
    ax2.set_xlabel('c/d',labelpad=-35)

    plt.ylim(np.percentile(spower,5),np.max(power))
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    plt.yscale('log')
    if title:
        plt.title(r'%s Power Spectrum' % name,y=1.01)
    ax1.legend()

def plot_log_pgram_oe(ax1,frequency_even,frequency_odd,power_even,power_odd,spower_even,spower_odd,name,min_p=1./24.,max_p=20.,title=False):

    ax1.plot(frequency_odd*11.57,power_odd,label='Raw Odd',color=colours[0],alpha=0.25)
    ax1.plot(frequency_odd*11.57,spower_odd,linewidth=3.0,label='Smoothed Odd',color=colours[0])

    ax1.plot(frequency_even*11.57,power_even,label='Raw Even',color=colours[2],alpha=0.25)
    ax1.plot(frequency_even*11.57,spower_even,linewidth=3.0,label='Smoothed Even',color=colours[2])

    ax1.set_xlim(1./max_p*11.57,1./min_p*11.57)
    ax1.set_xlabel(r'Frequency ($\mu$Hz)')
    ax1.set_ylabel(r'Power (ppm$^2/{\mu}Hz)$')

    ax2 = ax1.twiny()
    ax2.tick_params(axis="x",direction="in", pad=-20)

    ax2.set_xlim(1./max_p,1./min_p)
    ax2.set_xlabel('c/d',labelpad=-35)

    plt.ylim(np.percentile((spower_even),5),np.max(power_even))
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    plt.yscale('log')
    if title:
        plt.title(r'%s Power Spectrum' % name,y=1.01)
    ax1.legend()

def plot_hists(ax,data1,data2):
    all_data = np.concatenate((data1,data2))
    low, high = np.percentile(all_data,0.1),np.percentile(all_data,99.9)

    data_1 = data1[(data1>low)*(data1<high)]
    data_2 = data2[(data2>low)*(data2<high)]

    _, bins, _ = ax.hist(data_1, bins=30, alpha=0.5, density=True, color=colours[2]);

    ax.hist(data_2, bins=bins, alpha=0.5, density=True,color=colours[0]);
    ax.set_xlim()



def plot_info(ax, meta):
    vals = []
    keys = []
    for key,val in meta.items():
        keys.append(key+'\n')
        vals.append(str(val).replace('_',' ')+'\n')
    ax.text(0.0,0.83, ''.join(keys), size=20, va='top')
    ax.text(0.97,0.83,''.join(vals), size=20, va='top', ha='right')
    despine(ax=ax, left=True, bottom=True)
    setp(ax, xticks=[], yticks=[])

def plot_all(ts_kep,ts_tess,even,meta,save_file=None,formal_name='test',title=True):
    min_p,max_p=1./24.,20.

    PW,PH = 8.27, 11.69

    frequency_kep_odd, power_kep_odd, spower_kep_odd = get_pgram(ts_kep['time'][~even],ts_kep['flux'][~even],min_p=min_p,max_p=max_p)
    frequency_kep_even, power_kep_even, spower_kep_even = get_pgram(ts_kep['time'][even],ts_kep['flux'][even],min_p=min_p,max_p=max_p)

    frequency_tess, power_tess, spower_tess = get_pgram(ts_tess['time'],ts_tess['flux'],min_p=min_p,max_p=max_p)

    rc('axes', labelsize=7, titlesize=8)
    rc('font', size=6)
    rc('xtick', labelsize=7)
    rc('ytick', labelsize=7)
    rc('lines', linewidth=1)

    fig = plt.figure(figsize=(PW,PH))

    gs1 = GridSpec(2,2)
    gs1.update(top=0.95, bottom = 2/3.*1.02,hspace=0.25,left=0.09,right=0.96)
    gs2 = GridSpec(2,2)
    gs2.update(top=2/3.*0.98,bottom=1/3.*1.07,hspace=0.0,left=0.09,right=0.96)
    gs3 = GridSpec(2,2)
    gs3.update(top=1/3.*0.96,bottom=0.04,hspace=0.07,left=0.09,right=0.96)

    ax_lckepler_long = subplot(gs1[0,:])
    ax_info = subplot(gs1[1,0])
    ax_hists = subplot(gs1[1,1])

    ax_lckepler = subplot(gs2[0,:])
    ax_lctess= subplot(gs2[1,:])

    ax_pgkepler = subplot(gs3[0,:])
    ax_pgtess = subplot(gs3[1,:],sharex=ax_pgkepler)

    timerange = ts_tess['time'].max()-ts_tess['time'].min()
    t0 = np.nanmedian(ts_kep['time'])
    times = (ts_kep['time']>t0)*(ts_kep['time']<(t0+timerange))

    plot_lc(ax_lckepler_long,ts_kep['time'],ts_kep['flux'],formal_name)
    plot_info(ax_info,meta)
    plot_hists(ax_hists,ts_kep['flux'],ts_tess['flux'])

    plot_lc(ax_lckepler,ts_kep['time'][times],ts_kep['flux'][times],formal_name)
    plot_log_pgram_oe(ax_pgkepler,frequency_kep_even,frequency_kep_odd,power_kep_even,power_kep_odd,spower_kep_even,spower_kep_odd,formal_name)

    plot_lc(ax_lctess,ts_tess['time'],ts_tess['flux'],formal_name)
    plot_log_pgram(ax_pgtess,frequency_tess,power_tess,spower_tess,formal_name)

    if title:
        fig.suptitle(formal_name,y=0.99,fontsize=20)
    ax_pgkepler.set_title('Periodograms')

    if save_file is not None:
        plt.savefig(save_file)
