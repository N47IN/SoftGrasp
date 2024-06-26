#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Point_set.h>
#include <boost/iterator/function_output_iterator.hpp>

#include "include/utils.h"

// Typedefs.
using Kernel   = CGAL::Simple_cartesian<double>;
using FT       = Kernel::FT;
using Point_3  = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;

using Point_set  = CGAL::Point_set_3<Point_3>;
using Point_map  = typename Point_set::Point_map;
using Normal_map = typename Point_set::Vector_map;

using Neighbor_query = CGAL::Shape_detection::Point_set::K_neighbor_query_for_point_set<Point_set>;
using Region_type    = CGAL::Shape_detection::Point_set::Least_squares_cylinder_fit_region_for_point_set<Point_set>;
using Region_growing = CGAL::Shape_detection::Region_growing<Neighbor_query, Region_type>;

int main(int argc, char** argv) {

  // Load ply data either from a local folder or a user-provided file.
  const bool is_default_input = argc > 1 ? false : true;
  std::ifstream in(is_default_input ? CGAL::data_file_path("mug.ply") : argv[1]);

  CGAL::IO::set_ascii_mode(in);
  if (!in) {
    std::cerr << "ERROR: cannot read the input file!" << std::endl;
    return EXIT_FAILURE;
  }

  Point_set point_set;
  in >> point_set;
  in.close();
  std::cout << "* number of input points: " << point_set.size() << std::endl;
  assert(!is_default_input || point_set.size() == 1813);
  assert(point_set.has_normal_map()); // input should have normals

  // Default parameter values for the data file cuble.pwn.
  const std::size_t k               = 20;
  const FT          max_distance    = FT(1) / FT(100);
  const FT          max_angle       = FT(1);
  const std::size_t min_region_size = 1000;

  // Create instances of the classes Neighbor_query and Region_type.
  Neighbor_query neighbor_query = CGAL::Shape_detection::Point_set::make_k_neighbor_query(point_set, CGAL::parameters::k_neighbors(k));

  Region_type region_type = CGAL::Shape_detection::Point_set::make_least_squares_cylinder_fit_region(
    point_set,
    CGAL::parameters::
    maximum_distance(max_distance).
    maximum_angle(max_angle).
    minimum_region_size(min_region_size));

  // Create an instance of the region growing class.
  Region_growing region_growing(
    point_set, neighbor_query, region_type);

  // Add maps to get a colored output.
  Point_set::Property_map<unsigned char>
    red   = point_set.add_property_map<unsigned char>("red"  , 0).first,
    green = point_set.add_property_map<unsigned char>("green", 0).first,
    blue  = point_set.add_property_map<unsigned char>("blue" , 0).first;

  // Run the algorithm.
  CGAL::Random random;
  std::size_t num_cylinders = 0;
  region_growing.detect(
    boost::make_function_output_iterator(
      [&](const std::pair< Region_type::Primitive, std::vector<typename Point_set::Index> >& region) {

        // Assign a random color to each region.
        const unsigned char r = static_cast<unsigned char>(random.get_int(64, 192));
        const unsigned char g = static_cast<unsigned char>(random.get_int(64, 192));
        const unsigned char b = static_cast<unsigned char>(random.get_int(64, 192));

        for (auto id : region.second) {
          put(red, id, r);
          put(green, id, g);
          put(blue, id, b);
        }
        ++num_cylinders;
      }
    )
  );
  std::cout << "* number of found cylinders: " << num_cylinders << std::endl;
  assert(!is_default_input || num_cylinders == 2);

  // Save regions to a file.
  std::ofstream out("cylins.ply");
  CGAL::IO::set_ascii_mode(out);
  out << point_set;
  return EXIT_SUCCESS;
}
