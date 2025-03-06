#include <iostream>
#include <librealsense2/rs.hpp>
#include <zmq.hpp>

int main() {
    try {
        std::cout << "Initializing RealSense pipeline..." << std::endl;
        rs2::pipeline p;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
        p.start(cfg);

        zmq::context_t context(1);
		zmq::socket_t socket(context, ZMQ_PUB);
		socket.bind("tcp://*:5555");

		socket.set(zmq::sockopt::immediate, 1); 
		socket.set(zmq::sockopt::sndhwm, 1); 
		socket.set(zmq::sockopt::linger, 0);
		const int width = 640;
		const int height = 480;
		const int size = width * height * 3;

		std::cout << "Streaming video... Press Ctrl+C to exit." << std::endl;

		while (true) {
			rs2::frameset frames = p.wait_for_frames();
			rs2::frame color_frame = frames.get_color_frame();

			if (!color_frame) continue;

			std::vector<uint8_t> frame_data((uint8_t*)color_frame.get_data(), 
					(uint8_t*)color_frame.get_data() + size);
			zmq::message_t msg(color_frame.get_data(), size);
			socket.send(msg, zmq::send_flags::none);
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
