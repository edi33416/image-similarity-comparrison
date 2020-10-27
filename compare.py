import numpy as np
import cv2 as cv
import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle



# Compute PSNR from two images
# https://docs.opencv.org/master/d5/dc4/tutorial_video_input_psnr_ssim.html
def get_psnr(I1, I2):
    s1 = cv.absdiff(I1, I2) #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 * s1            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 1e-10:        # sum channels
        return 0            # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

# Compute MSSISM from two images
# https://docs.opencv.org/master/d5/dc4/tutorial_video_input_psnr_ssim.html
def get_mssism(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv.divide(t3, t1)    # ssim_map =  t3./t1;
    mssim = cv.mean(ssim_map)       # mssim = average of ssim map
    return mssim

def get_raw_filename(filename):
    base_filename = os.path.basename(filename)
    return os.path.splitext(base_filename)[0]

def get_results(ground_truth, results_dirs):
    if os.path.isfile("results.pickle"):
        r = pickle.load(open("results.pickle", "rb"))
        return r

    comp_dict = {}
    for i in os.listdir(ground_truth):
        gt_base = "normal"
        img_key = get_raw_filename(i)[len(gt_base):]
        if img_key in comp_dict:
            assert 0, "Key already present"
        comp_dict[img_key] = {
                "gt": ground_truth + "/" + i,
                "comp": []
        }

    for res_dir in results_dirs:
        for i in os.listdir(res_dir):
            img_base = "low"
            img_key = get_raw_filename(i)[len(img_base):]

            comp_dict[img_key]["comp"].append({
                "method_name": res_dir,
                "path": res_dir + "/" + i
            })

    psnr_result = {}
    mssism_result = {}
    methods = []
    labels = []
    for i in comp_dict:
        labels.append(i)
        gt = cv.imread(comp_dict[i]['gt'])
        for comp in comp_dict[i]["comp"]:
            if comp["method_name"] not in methods:
                methods.append(comp["method_name"])

            img = cv.imread(comp['path'])

            if comp["method_name"] not in psnr_result:
                psnr_result[comp["method_name"]] = []
            psnr_result[comp["method_name"]].append(get_psnr(gt, img))

            if comp["method_name"] not in mssism_result:
                mssism_result[comp["method_name"]] = []
            r, g, b, _ = get_mssism(gt, img)
            mssism_result[comp["method_name"]].append((r + g + b) / 3)

    r = {}
    r["psnr_result"] = psnr_result
    r["mssism_result"] = mssism_result
    r["methods"] = methods
    r["labels"] = labels

    pickle.dump(r, open("results.pickle", "wb"))

    return r

def plot_result(result, methods, xlabel, ylabel, fig_title, fig_name,
        slice_size = None, with_mean = False):

    for m in methods:
        if slice_size is None:
            slice_size = len(result[m])
        else:
            slice_size = min(slice_size, len(result[m]))

        #p = plt.plot(result[m][:slice_size], 'o', label = m, markersize = 5)
        p = plt.plot(result[m][:slice_size], 'o', label = m)

        if with_mean:
            mean_val = np.mean(result[m][:slice_size])
            res_mean = [mean_val for x in result[m][:slice_size]]
            plt.plot(res_mean, color = p[0].get_color(), label = f"{m} mean value = {mean_val:.2f}")
        else:
            res_median_hlp = [x for x in result[m][:slice_size]]
            res_median_hlp.sort()
            median_val = res_median_hlp[int(len(res_median_hlp) / 2)]
            res_median = [median_val for x in res_median_hlp]
            plt.plot(res_median, color = p[0].get_color(), label = f"{m} median value = {median_val:.2f}")

    lg = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    tt = plt.suptitle(fig_title)
    plt.savefig(fig_name, dpi=300, bbox_extra_artists=(lg, tt,), bbox_inches='tight')
    plt.clf()

def plot_barchart_result(result, labels, methods, xlabel, ylabel, fig_title, fig_name,
        slice_size = None, with_mean = False):
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    x = np.arange(len(labels))
    width = 3
    fig, ax = plt.subplots()
    for m in methods:
        ax.bar(x - width / 2, result[m], width, label = m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(fig_name)
    plt.clf()

def main():

    parser = argparse.ArgumentParser(description="Plot PSNR and MSSISM charts for low light image processing methods")
    parser.add_argument("-gt", "--ground_truth", default="./gt/",
            help="Path to ground truth images")
    parser.add_argument("-res", "--results", nargs='+',
            default=["./EnlightenGAN/", "./RetinexNET/", "./Zero-DCE/", "./res-Costin/"],
            help="Path to directories of results to be comapred with gt")
    parser.add_argument("-s", "--slice_size", type=int, default=10,
            help="This script will also plot a (smaller) subset equal to the slice size")
    args = parser.parse_args()

    r = get_results(args.ground_truth, args.results)
    psnr_result = r["psnr_result"]
    mssism_result = r["mssism_result"]
    methods = r["methods"]
    labels = r["labels"]

    slice_size = args.slice_size

    plot_result(result = psnr_result,
            methods = methods,
            xlabel = "Input Image Index",
            ylabel = "PSNR Score",
            fig_title = "PSNR - Lower is better",
            fig_name = "psnr-median-comparison.png")

    plot_result(result = psnr_result,
            methods = methods,
            xlabel = "Input Image Index",
            ylabel = "PSNR Score",
            fig_title = "PSNR - Lower is better",
            fig_name = "psnr-mean-comparison.png",
            with_mean = True)

    plot_result(result = psnr_result,
            methods = methods,
            xlabel = "Input Image Index",
            ylabel = "PSNR Score",
            fig_title = f"PSNR - {slice_size} Imgs - Lower is better",
            fig_name = f"psnr-{slice_size}-median-comparison.png",
            slice_size = slice_size)

    plot_result(result = psnr_result,
            methods = methods,
            xlabel = "Input Image Index",
            ylabel = "PSNR Score",
            fig_title = f"PSNR - {slice_size} Imgs - Lower is better",
            fig_name = f"psnr-{slice_size}-mean-comparison.png",
            slice_size = slice_size,
            with_mean = True)

    print(f"PSNR {slice_size} has labels: {labels[:slice_size]}")

    plot_result(result = mssism_result,
            methods = methods,
            xlabel = "Input Image Index",
            ylabel = "MSSISM Score",
            fig_title = "MSSISM - Higher is better",
            fig_name = "mssism-median-comparison.png")

    plot_result(result = mssism_result,
            methods = methods,
            xlabel = "Input Image Index",
            ylabel = "MSSISM Score",
            fig_title = "MSSISM - Higher is better",
            fig_name = "mssism-mean-comparison.png",
            with_mean = True)

    plot_result(result = mssism_result,
            methods = methods,
            xlabel = "Input Image Index",
            ylabel = "MSSISM Score",
            fig_title = f"MSSISM - {slice_size} Imgs - Higher is better",
            fig_name = f"mssism-{slice_size}-median-comparison.png",
            slice_size = slice_size)

    plot_result(result = mssism_result,
            methods = methods,
            xlabel = "Input Image Index",
            ylabel = "MSSISM Score",
            fig_title = f"MSSISM - {slice_size} Imgs - Higher is better",
            fig_name = f"mssism-{slice_size}-mean-comparison.png",
            slice_size = slice_size,
            with_mean = True)

if __name__ == "__main__":
    main()
