#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>
#include <CGAL/boost/graph/helpers.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Triangle = std::array<std::size_t, 3>;
using Mesh = CGAL::Surface_mesh<Point>;

struct PlyCounts {
  std::size_t vertices = 0;
  std::size_t faces = 0;
};

PlyCounts read_ply_counts(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Could not open input: " + path.string());
  }

  PlyCounts counts;
  std::string line;
  bool saw_header_end = false;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    std::istringstream fields(line);
    std::string keyword;
    fields >> keyword;
    if (keyword == "element") {
      std::string kind;
      std::size_t count = 0;
      fields >> kind >> count;
      if (kind == "vertex") {
        counts.vertices = count;
      } else if (kind == "face") {
        counts.faces = count;
      }
    } else if (keyword == "end_header") {
      saw_header_end = true;
      break;
    }
  }
  if (!saw_header_end) {
    throw std::runtime_error("Input does not contain a complete PLY header");
  }
  return counts;
}

std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (const char character : value) {
    switch (character) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << character; break;
    }
  }
  out << '"';
  return out.str();
}

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 6) {
      std::cerr
          << "Usage: " << argv[0]
          << " INPUT.ply OUTPUT.ply ALPHA [OFFSET] [REPORT.json]\n";
      return 2;
    }

    const std::filesystem::path input_path = argv[1];
    const std::filesystem::path output_path = argv[2];
    const double alpha = std::stod(argv[3]);
    const double offset = argc >= 5 ? std::stod(argv[4]) : alpha / 30.0;
    const std::filesystem::path report_path =
        argc >= 6 ? std::filesystem::path(argv[5])
                  : output_path.parent_path() /
                        (output_path.stem().string() + ".json");
    if (!(alpha > 0.0) || !(offset > 0.0)) {
      throw std::runtime_error("ALPHA and OFFSET must be positive");
    }

    const auto started = std::chrono::steady_clock::now();
    const PlyCounts header_counts = read_ply_counts(input_path);
    std::vector<Point> points;
    std::vector<Triangle> triangles;
    points.reserve(header_counts.vertices);
    triangles.reserve(header_counts.faces);

    std::cerr << "[read] expected vertices=" << header_counts.vertices
              << " faces=" << header_counts.faces << std::endl;
    if (!CGAL::IO::read_polygon_soup(
            input_path.string(), points, triangles,
            CGAL::parameters::verbose(true))) {
      throw std::runtime_error("CGAL could not read the polygon soup");
    }
    if (points.empty() || triangles.empty()) {
      throw std::runtime_error("Input polygon soup is empty");
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    for (const Point& point : points) {
      min_x = std::min(min_x, point.x());
      min_y = std::min(min_y, point.y());
      min_z = std::min(min_z, point.z());
      max_x = std::max(max_x, point.x());
      max_y = std::max(max_y, point.y());
      max_z = std::max(max_z, point.z());
    }
    const double diagonal = std::sqrt(
        std::pow(max_x - min_x, 2) + std::pow(max_y - min_y, 2) +
        std::pow(max_z - min_z, 2));
    std::cerr << std::setprecision(10)
              << "[input] vertices=" << points.size()
              << " triangles=" << triangles.size()
              << " diagonal=" << diagonal << " alpha=" << alpha
              << " offset=" << offset << std::endl;

    Mesh wrap;
    CGAL::alpha_wrap_3(points, triangles, alpha, offset, wrap);
    if (wrap.is_empty()) {
      throw std::runtime_error("Alpha Wrapping returned an empty mesh");
    }

    const bool valid = CGAL::is_valid_polygon_mesh(wrap);
    const bool closed = CGAL::is_closed(wrap);
    const bool triangulated = CGAL::is_triangle_mesh(wrap);
    auto component_map =
        wrap.add_property_map<Mesh::Face_index, std::size_t>("f:component", 0)
            .first;
    const std::size_t components = PMP::connected_components(wrap, component_map);

    std::filesystem::create_directories(output_path.parent_path());
    if (!CGAL::IO::write_polygon_mesh(
            output_path.string(), wrap,
            CGAL::parameters::use_binary_mode(true).stream_precision(17))) {
      throw std::runtime_error("CGAL could not write the wrapped mesh");
    }

    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    std::ostringstream report;
    report << std::boolalpha << std::setprecision(17)
           << "{\n"
           << "  \"input\": " << json_string(input_path.string()) << ",\n"
           << "  \"output\": " << json_string(output_path.string()) << ",\n"
           << "  \"input_vertices\": " << points.size() << ",\n"
           << "  \"input_triangles\": " << triangles.size() << ",\n"
           << "  \"bounding_box_diagonal\": " << diagonal << ",\n"
           << "  \"alpha\": " << alpha << ",\n"
           << "  \"offset\": " << offset << ",\n"
           << "  \"output_vertices\": " << num_vertices(wrap) << ",\n"
           << "  \"output_edges\": " << num_edges(wrap) << ",\n"
           << "  \"output_faces\": " << num_faces(wrap) << ",\n"
           << "  \"connected_components\": " << components << ",\n"
           << "  \"valid_polygon_mesh\": " << valid << ",\n"
           << "  \"closed\": " << closed << ",\n"
           << "  \"triangle_mesh\": " << triangulated << ",\n"
           << "  \"elapsed_seconds\": " << elapsed << "\n"
           << "}\n";
    std::filesystem::create_directories(report_path.parent_path());
    std::ofstream report_file(report_path);
    report_file << report.str();
    if (!report_file) {
      throw std::runtime_error("Could not write report: " + report_path.string());
    }
    std::cout << report.str();
    return valid && closed && triangulated ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }
}
