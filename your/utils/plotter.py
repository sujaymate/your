import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import numpy as np
import pylab as plt
from matplotlib import gridspec
from scipy.signal import detrend

from your.utils.math import smad_plotter


def plot_h5(
    h5_file,
    save=True,
    detrend_ft=True,
    publication=False,
    mad_filter=False,
    outdir=None,
):
    """
    Plot the h5 candidates

    Args:
        mad_filter (int): use MAD filter to clip data
        h5_file (str): Name of the h5 file
        save (bool): Save the file as a png
        detrend_ft (bool): detrend the frequency time plot
        publication (bool): make publication quality plot
        outdir (str): Path to the save the files into.

    Returns:
        None

    """
    with h5py.File(h5_file, "r") as f:
        dm_time = np.array(f["data_dm_time"])
        if detrend_ft:
            freq_time = detrend(np.array(f["data_freq_time"])[:, ::-1].T)
        else:
            freq_time = np.array(f["data_freq_time"])[:, ::-1].T
        dm_time[dm_time != dm_time] = 0
        freq_time[freq_time != freq_time] = 0
        freq_time -= np.median(freq_time)
        freq_time /= np.std(freq_time)
        fch1, foff, nchan, dm, cand_id, tsamp, dm_opt, snr, snr_opt, width = (
            f.attrs["fch1"],
            f.attrs["foff"],
            f.attrs["nchans"],
            f.attrs["dm"],
            f.attrs["cand_id"],
            f.attrs["tsamp"],
            f.attrs["dm_opt"],
            f.attrs["snr"],
            f.attrs["snr_opt"],
            f.attrs["width"],
        )
        
        # Get DM vs SNR and nearby events data if it exists
        add_verification = False
        cluster_dir = Path(h5_file).parent.parent.joinpath("cluster_cand")
        if cluster_dir.exists():
            add_verification = True
            
            cluster_h5file = sorted(cluster_dir.glob("*.h5"))[0]
            with h5py.File(cluster_h5file, 'r') as cluster_file:
                det_events = cluster_file['det_events'][()]

            DMvsSNR = get_DM_vs_SNR(det_events, f.attrs['label'])
            nearby_events = get_nearby_events(det_events, f.attrs['tcand'])


        tlen = freq_time.shape[1]
        if tlen != 256:
            logging.warning(
                "Lengh of time axis is not 256. This data is probably not pre-processed."
            )
        l = np.linspace(-tlen // 2, tlen // 2, tlen)
        if width > 1:
            ts = l * tsamp * width * 1000 / 2
        else:
            ts = l * tsamp * 1000

        if mad_filter:
            freq_time = smad_plotter(freq_time, float(mad_filter))

        plt.clf()

        if publication:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 7), sharex="col")

        else:
            fig = plt.figure(figsize=(12, 9))
            gs = gridspec.GridSpec(3, 2, width_ratios=[2, 2], height_ratios=[1, 1, 1])
            ax2 = plt.subplot(gs[1, 0])
            ax1 = plt.subplot(gs[0, 0], sharex=ax2)
            ax3 = plt.subplot(gs[2, 0], sharex=ax2)
            ax4 = plt.subplot(gs[0, 1])
            if add_verification:
                ax5 = plt.subplot(gs[1, 1])
                ax6 = plt.subplot(gs[2, 1])

            # print text
            to_print = [f"File: {f.attrs['basename']}\n",
                        f"Beam: {f.attrs['beam']:04d}\n",
                        f"Arrival Time (UTC): {f.attrs['tcand_utc']}\n",
                        f"Rel. Arrival Time (s): {f.attrs['tcand']: 7.2f}\n",
                        f"Boxcar width (nsamples): {f.attrs['width']:d}\n",
                        f"Boxcar width (s): {f.attrs['width']*f.attrs['tsamp']: 5.3f}\n",
                        f"DM (pc cm$^{{-3}}$): {f.attrs['dm']: 6.1f}\n",
                        f"SNR: {f.attrs['snr']: 6.2f}\n",
                        f"RA (deg): {f.attrs['ra_deg']: 6.2f}\n",
                        f"Dec (Deg): {f.attrs['dec_deg']: 6.2f}"]
            str_print = "".join(to_print)
            ax4.text(0.0, 0., str_print, fontsize=12, ha="left", va="bottom", wrap=True)
            ax4.axis("off")

        ax1.plot(ts, freq_time.sum(0), "k-")
        ax1.set_ylabel("Flux (Arb. Units)")
        ax2.imshow(
            freq_time,
            aspect="auto",
            extent=[ts[0], ts[-1], fch1 + (nchan * foff), fch1], # Changed this to flip frequency axis ARVIND
            interpolation="none",
            origin="lower",
        )
        ax2.set_ylabel("Frequency (MHz)")
        ax3.imshow(
            dm_time,
            aspect="auto",
            extent=[ts[0], ts[-1], 2 * dm, 0],
            interpolation="none",
        )
        ax3.set_ylabel(r"DM (pc cm$^{-3}$)")
        ax3.set_xlabel("Time (ms)")
        
        if add_verification:
            ax5.scatter(DMvsSNR['DM'], DMvsSNR['SNR'], c='C0')
            ax5.set_xlabel(r"DM (pc cm$^{-3}$)")
            # DM at max SNR
            DM_max = DMvsSNR['DM'][(np.argmax(DMvsSNR['SNR']))]
            ax5.set_xlim([DM_max - 75, DM_max + 75])
            ax5.set_ylabel("SNR")
            
            # this cluster
            cluster = nearby_events['Label'] == f.attrs['label']
            nearby_events['TIME'] = nearby_events['TIME'] - f.attrs['tcand']
            ax6.scatter(nearby_events['TIME'][~cluster], nearby_events['DM'][~cluster], s=nearby_events['SNR'][~cluster],
                        facecolors='none', edgecolors='grey')
            ax6.scatter(nearby_events['TIME'][cluster], nearby_events['DM'][cluster], s=nearby_events['SNR'][cluster],
                        facecolors='none', edgecolors='C0')
            ax6.set_xlabel("Time (s)")
            ax6.set_ylabel(r"DM (pc cm$^{-3}$)")
            ax6.set_xlim([-50, 50])
            ax6.set_ylim([0, 3000])
            

        plt.tight_layout()
        if save:
            if outdir:
                filename = outdir + os.path.basename(h5_file)[:-3] + ".png"
            else:
                filename = h5_file[:-3] + ".png"
            plt.savefig(filename, bbox_inches="tight") # Removed dpi=300 to reduce file size ARVIND
            plt.close(fig)
        else:
            plt.close()

    return None


def save_bandpass(
    your_object, bandpass, chan_nos=None, mask=None, outdir=None, outname=None
):
    """
    Plots and saves the bandpass

    Args:
        your_object: Your object
        bandpass (np.ndarray): Bandpass of the data
        chan_nos (np.ndarray): Array of channel numbers
        mask (np.ndarray): Boolean Array of channel mask
        outdir (str): Output directory to save the plot
        outname (str): Name of the bandpass file
    """

    freqs = your_object.chan_freqs
    foff = your_object.your_header.foff

    if not outdir:
        outdir = "./"

    if chan_nos is None:
        chan_nos = np.arange(0, bandpass.shape[0])

    if not outname:
        bp_plot = outdir + your_object.your_header.basename + "_bandpass.png"
    else:
        bp_plot = outname

    fig = plt.figure()
    ax11 = fig.add_subplot(111)
    if foff < 0:
        ax11.invert_xaxis()

    ax11.plot(freqs, bandpass, "k-", label="Bandpass")
    if mask is not None:
        if mask.sum():
            logging.info("Flagged %d channels", mask.sum())
            ax11.plot(freqs[mask], bandpass[mask], "r.", label="Flagged Channels")
    ax11.set_xlabel("Frequency (MHz)")
    ax11.set_ylabel("Arb. Units")
    ax11.legend()

    ax21 = ax11.twiny()
    ax21.plot(chan_nos, bandpass, alpha=0)
    ax21.set_xlabel("Channel Numbers")

    return plt.savefig(bp_plot, bbox_inches="tight", dpi=300)


def get_DM_vs_SNR(
    det_events, label
):
    """ Function to return DM and SNR slice for given candidate from detected events

    Args:
        det_events (np.recarray): Detected singlepulse events
        label (int): Cluster label for the candidate

    Returns:
        np.recarray : Detected event DM and SNR slice for the cluster in which the candidate belongs
    """

    cluster = det_events['Label'] == label
    return det_events[['DM', 'SNR']][cluster]


def get_nearby_events(
    det_events, tcand
):
    """ Function to return events in the +/- 50 seconds of the detected candidate.

    Args:
        det_events (np.recarray): Detected singlepulse events
        tcand (float): Candidate time relative to the start of filterbank file

    Returns:
        np.recarray: Detected event slice
    """
    
    time_slice = (det_events['TIME'] > (tcand - 50)) & (det_events['TIME'] < (tcand + 50))
    return det_events[time_slice]