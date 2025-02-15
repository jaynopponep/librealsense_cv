#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>  // OpenCV for display

int main() {
    try {
        std::cout << "Initializing RealSense pipeline..." << std::endl;
        rs2::pipeline p;
        p.start();

        std::cout << "Streaming video... Press 'q' to exit." << std::endl;

        while (true) {
            rs2::frameset frames = p.wait_for_frames();
            rs2::frame color_frame = frames.get_color_frame();

            // Get frame dimensions
            const int width = color_frame.as<rs2::video_frame>().get_width();
            const int height = color_frame.as<rs2::video_frame>().get_height();

            // Convert RealSense frame to OpenCV matrix
            cv::Mat color_image(cv::Size(width, height), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

            // Show the image
            cv::imshow("RealSense Camera Stream", color_image);

            // Exit if 'q' is pressed
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        std::cout << "Stopping pipeline..." << std::endl;
        p.stop();
    } catch (const rs2::error &e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << "General error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}

