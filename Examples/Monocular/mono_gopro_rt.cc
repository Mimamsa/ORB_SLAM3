/*

Usage:
    Mapping:
./mono_gopro_rt --save_map ./atlas.osa\
               --enable_gui\
               --mask_img ./slam_mask.png\
               --max_lost_frames 60
    Tracking:
./mono_gopro_rt --load_map ./atlas.osa\
               --enable_gui\
               --mask_img ./slam_mask.png
 */
#include <fstream>
#include <iostream>
#include <chrono>
#include <signal.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <CLI11.hpp>

#include <System.h>


using namespace std;


bool b_continue_session;

void exit_loop_handler(int s) {
   cout << "Finishing session" << endl;
   b_continue_session = false;

}

int main(int argc, char **argv) {

    // CLI parsing
    CLI::App app{"GoPro Hero12 + Elato HD60 X realtime SLAM"};

    std::string vocabulary = "../../Vocabulary/ORBvoc.txt";
    app.add_option("-v,--vocabulary", vocabulary)->capture_default_str();

    std::string settings = "gopro12_maxlens_fisheye_setting.yaml";
    app.add_option("-s,--settings", settings)->capture_default_str();

    std::string stream_device = "/dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB35125XOLJ-video-index0";
    app.add_option("-i, --stream_device", stream_device)->capture_default_str();

    std::string load_map;
    app.add_option("-l,--load_map", load_map);

    std::string save_map;
    app.add_option("--save_map", save_map);

    bool enable_gui = false;
    app.add_flag("-g,--enable_gui", enable_gui);

    std::string mask_img_path;
    app.add_option("--mask_img", mask_img_path);

    std::string output_trajectory_csv;
    app.add_option("-o,--output_trajectory_csv", output_trajectory_csv);

    // Aruco tag for initialization
    int aruco_dict_id = cv::aruco::DICT_4X4_50;
    app.add_option("--aruco_dict_id", aruco_dict_id);

    int init_tag_id = 13;
    app.add_option("--init_tag_id", init_tag_id);

    float init_tag_size = 0.16; // in meters
    app.add_option("--init_tag_size", init_tag_size);


    // if lost more than max_lost_frames, terminate
    // disable the check if <= 0
    int max_lost_frames = -1;
    app.add_option("--max_lost_frames", max_lost_frames);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    // define SIG_INT behavior
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
    b_continue_session = true;

    cv::setNumThreads(4);

    // open settings to get image resolution
    cv::FileStorage fsSettings(settings, cv::FileStorage::READ);
    if(!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << settings << endl;
        exit(-1);
    }
    cv::Size img_size(fsSettings["Camera.width"],fsSettings["Camera.height"]);
    fsSettings.release();

    // Create SLAM system. It initializes all system threads and gets ready to
    // process frames.
    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::aruco::getPredefinedDictionary(aruco_dict_id);
    ORB_SLAM3::System SLAM(
        vocabulary,
        settings,
        ORB_SLAM3::System::MONOCULAR,
        enable_gui, load_map, save_map,
        aruco_dict, init_tag_id, init_tag_size);

    cv::VideoCapture cap(stream_device, cv::CAP_V4L2);
    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // support size see: https://help.elgato.com/hc/en-us/articles/360027952992-Supported-resolutions-for-Elgato-Game-Capture-HD
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);

    double fps = cap.get(cv::CAP_PROP_FPS);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);    
    std::cout << "video FPS " << fps << std::endl;
    std::cout << "video height " << height << std::endl;
    std::cout << "video width " << width << std::endl;

    // load mask image
    cv::Mat mask_img;
    if (!mask_img_path.empty()) {
        mask_img = cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);
        if (mask_img.size() != img_size) {
            std::cout << "Mask img size mismatch! Converting " << mask_img.size() << " to " << img_size << std::endl;
            cv::resize(mask_img, mask_img, img_size);
        }
    }

    // define frame ROI
    int x_start = (1280-960)/2;
    cv::Rect myROI(x_start, 0, 960, 720);
    cv::Mat cropped;

    int frame_idx = 0;
    int n_lost_frames = 0;
    // Main loop
    while (b_continue_session) {
        double tframe = (double)frame_idx / fps;
    
        // read frame from video
        cv::Mat im, im_track;
        bool success = cap.read(im); // or cap >> frame;
        if (!success) {
            std::cout << "cap.read failed!\n" << std::endl;
            break;
        }

        // slice frame from (1280,720) to (960,720)
        // https://stackoverflow.com/questions/8267191/how-to-crop-a-cvmat-in-opencv
        cv::Mat croppedRef(im, myROI);
        croppedRef.copyTo(im);

        // resize image and draw gripper mask
        im_track = im.clone();
        if (im_track.size() != img_size) {
            cv::resize(im_track, im_track, img_size);
        }

        // apply mask image if loaded
        if (!mask_img.empty()) {
            im_track.setTo(cv::Scalar(0,0,0), mask_img);
        }

        // cv::imshow("live", im);
        // if (cv::waitKey(1) == 'q') {
        //     break;
        // }

        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();

        auto result = SLAM.LocalizeMonocular(im_track, tframe);

        // check lost frames
        if (! result.second){
            n_lost_frames += 1;
            std::cout << "n_lost_frames=" << n_lost_frames << std::endl;
        }
        if ((max_lost_frames > 0) && (n_lost_frames >= max_lost_frames)){
            std::cout << "Lost tracking on " << n_lost_frames << " >= " << max_lost_frames << " frames. Terminating!" << std::endl;
            SLAM.Shutdown();
            return 1;
        }

        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();

        double ttrack =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
                .count();

        if (frame_idx % 100 == 0) {
            std::cout<<"Video FPS: "<<fps<<"\n";
            std::cout<<"ORB-SLAM 3 running at: "<<1./ttrack<< " FPS\n";
        }
        frame_idx++;
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    if (!output_trajectory_csv.empty()) {
        SLAM.SaveTrajectoryCSV(output_trajectory_csv);
    }

    return 0;
}
