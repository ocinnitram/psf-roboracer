#pragma once

#include <memory>
#include <string>
#include <vector>

namespace sf {

struct Point2D {
    double x = 0.0;
    double y = 0.0;
};

struct Frenet {
    double s = 0.0;
    double n = 0.0;
    double alpha = 0.0;
};

struct TrackData {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> s;
    std::vector<double> s_norm;
    std::vector<double> track_yaw;
    std::vector<double> kappa;
    std::vector<double> kappa_s;
    std::vector<double> w_l;
    std::vector<double> w_r;
    std::vector<double> n_x;
    std::vector<double> n_y;
};

class TrackHandler {
public:
    TrackHandler();
    ~TrackHandler();

    TrackHandler(const TrackHandler&) = delete;
    TrackHandler& operator=(const TrackHandler&) = delete;
    TrackHandler(TrackHandler&&) = delete;
    TrackHandler& operator=(TrackHandler&&) = delete;

    bool load(const std::string& csv_path);
    const TrackData& data() const { return m_data; }
    Frenet to_frenet(double x, double y, double yaw) const;

    int nearest_index(double x, double y) const;
    double sample_at_s(const std::vector<double>& values, double s) const;
    double kappa_s(double s) const;
    double width_left(double s) const;
    double width_right(double s) const;
    double length() const { return m_length; }

private:
    struct Index;

    TrackData m_data;
    double m_length = 0.0;
    std::unique_ptr<Index> m_index;
};

} // namespace sf
