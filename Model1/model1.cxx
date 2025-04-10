#include <array>
#include <boost/math/constants/constants.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

static constexpr double electron_rest_energy_ev = 510998.9461;
static constexpr double elementary_charge = 1.60217662e-19;
static constexpr double vacuum_permittivity = 8.8541878128e-12;
static constexpr double electron_mass = 9.1093837015e-31;
static constexpr double hbar_ev = 6.582119569e-16; // eV * s
static constexpr double c_light = 299792458;
static constexpr double constant = 0.013595738304; // sqrt(e^2 / (16 pi^3 epsilon_0 hbar c))

class Parameters {
public:
  Parameters(const char* input_path, boost::mpi::communicator& world);

  // Beam Parameters
  double cutoff;
  double mu_x;
  double mu_y;
  double sigma_x;
  double sigma_y;
  double sigma_z;
  double emit_n_x;
  double emit_n_y;
  double beam_energy_initial_ev;
  double percent_energy_spread;

  // Plasma Parameters
  double plasma_length;
  double plasma_density;
  double accelerating_field;
  double ion_charge_state;

  // Radiation Parameters
  double energy_min;
  double energy_max;
  std::size_t energy_num;
  double phix_min;
  double phix_max;
  std::size_t phix_num;
  double phiy_min;
  double phiy_max;
  std::size_t phiy_num;
  std::string energy_scale;

  // Non-Physcial Parameters
  unsigned seed;
  std::size_t steps;
  std::size_t particles_target;

  // Simulation Options
  std::string solver;
  bool output_trajectories;
  std::string general_output_filename;
  std::string radiation_output_filename;
  std::string particles_output_filename;

  // Computed Options
  double plasma_frequency;
  double plasma_angular_wavenumber;
  double plasma_skin_depth;
  double gamma_initial;
  double time_step;
  double z_step;
  std::size_t compute_processes;
  std::size_t actual_particles;
  std::size_t particles_per_process;

  void writeOutputFile(double seconds);

private:

  void computeDependentParameters(int processes);

};

static std::optional<std::string> readFile(const char* path)
// Reads the file located at 'path' into a string. If there were any errors
// opening or reading the file, an empty optional is returned. Otherwise, an
// optional containing the string is returned.
{
  assert(path);
  try {
    std::ifstream file;
    file.exceptions(file.failbit | file.badbit);
    file.open(path);
    file.seekg(0, file.end);
    auto file_size = boost::numeric_cast<std::size_t>(std::streamoff(file.tellg()));
    file.seekg(0, file.beg);
    std::string result;
    result.resize(file_size);
    file.read(&result[0], file_size);
    return result;
  } catch (const std::ifstream::failure&) {
    return std::nullopt;
  }
}

static std::map<std::string, std::string> inputToDict(std::string input)
// Reads a string containing a number of lines of the form <key> = <value> into
// a dictionary mapping keys to values. The exact grammar is:
// inputfile      := { [ key-value-pair ] , "\n" } , [ key-value-pair ] , EOF ;
// key-value-pair := word , "=" , word ;
// word           := { space } , { character } , { space } ;
// space          := " " | "\t" | "\v" | "\f" | "\r" ;
// character      := ( byte - space - "\n" - "=") ;
{
  std::map<std::string, std::string> dict;
  auto iterator = input.begin();

  // iterate through the lines
  while (true) {

    // find start of key
    while (true) {
      if (iterator == input.end())
        return dict;
      if (*iterator == '=')
        throw std::runtime_error("bad input file: expected key before =");
      if (!std::isspace(*iterator))
        break;
      ++iterator;
    }
    auto key_begin = iterator;
    ++iterator;

    // find end of key
    while (true) {
      if (iterator == input.end() || *iterator == '\n')
        throw std::runtime_error("bad input file: expected = after key");
      if (std::isspace(*iterator) || *iterator == '=')
        break;
      ++iterator;
    }
    auto key_end = iterator;
    while (*iterator != '=') {
      if (iterator == input.end() || *iterator == '\n' || !std::isspace(*iterator))
        throw std::runtime_error("bad input file: expected = after key");
      ++iterator;
    }
    ++iterator;

    // find start of value
    while (true) {
      if (iterator == input.end() || *iterator == '\n')
        throw std::runtime_error("bad input file: expected value after =");
      if (*iterator == '=')
        throw std::runtime_error("bad input file: more than one = on line");
      if (!std::isspace(*iterator))
        break;
      ++iterator;
    }
    auto value_begin = iterator;

    // find end of value
    while (true) {
      if (iterator == input.end() || std::isspace(*iterator))
        break;
      if (*iterator == '=')
        throw std::runtime_error("bad input file: = in value");
      ++iterator;
    }
    auto value_end = iterator;

    // add value to dictionary
    std::string key{key_begin, key_end};
    std::string value{value_begin, value_end};
    if (dict.find(key) != dict.end())
      throw std::runtime_error("bad input file: duplicate keys");
    dict[key] = value;

    // advance to end of line
    while (true) {
      if (iterator == input.end())
        break;
      if (*iterator == '\n') {
        ++iterator;
        break;
      }
      if (!std::isspace(*iterator))
        throw std::runtime_error("bad input file: line must end after value");
      ++iterator;
    }

  }
}

static void get_from_dict(std::map<std::string, std::string>& dict,
  std::string name, std::string& value)
{
  auto value_iter = dict.find(name);
  if (value_iter == dict.end())
    throw std::runtime_error(std::string("parameter ") + name +
      " not in input file");
  value = value_iter->second;
  dict.erase(value_iter);
}

static void get_from_dict(std::map<std::string, std::string>& dict,
  std::string name, double& value)
{
  std::string value_str;
  get_from_dict(dict, std::move(name), value_str);
  try {
    value = std::stod(value_str);
  } catch (const std::logic_error&) {
    throw std::runtime_error(std::string("unable to convert value of parameter "
      ) + name + " to real number");
  }
}

template <typename T>
static std::enable_if_t<std::is_unsigned_v<T>, void> get_from_dict(
  std::map<std::string, std::string>& dict, std::string name, T& value)
{
  std::string value_str;
  get_from_dict(dict, std::move(name), value_str);
  try {
    value = boost::numeric_cast<T>(std::stoull(value_str));
  } catch (const std::logic_error&) {
    throw std::runtime_error(std::string("unable to convert value of parameter "
    ) + name + " to (unsigned) integer");
  } catch (boost::numeric::bad_numeric_cast&) {
    throw std::runtime_error(std::string("value of parameter ") + name +
      "is too large for c++ type of corresponding parameter");
  }
}

static void get_from_dict(std::map<std::string, std::string>& dict,
  std::string name, bool& value)
{
  std::string value_str;
  get_from_dict(dict, std::move(name), value_str);
  if (value_str == "true" || value_str == "True")
    value = true;
  else if (value_str == "false" || value_str == "False")
    value = false;
  else
    throw std::runtime_error(std::string("value of parameter ") + name +
      " must be either true or false");
}

#define READ_PARAMETER(dict, param) get_from_dict(dict, #param, param)

Parameters::Parameters(const char* input_path, boost::mpi::communicator& world)
{
  auto input = readFile(input_path);
  if (!input)
    throw std::runtime_error("unable to open input");
  auto dict = inputToDict(*input);
  READ_PARAMETER(dict, cutoff);
  READ_PARAMETER(dict, mu_x);
  READ_PARAMETER(dict, mu_y);
  READ_PARAMETER(dict, sigma_x);
  READ_PARAMETER(dict, sigma_y);
  READ_PARAMETER(dict, emit_n_x);
  READ_PARAMETER(dict, emit_n_y);
  READ_PARAMETER(dict, beam_energy_initial_ev);
  READ_PARAMETER(dict, percent_energy_spread);
  READ_PARAMETER(dict, plasma_length);
  READ_PARAMETER(dict, plasma_density);
  READ_PARAMETER(dict, accelerating_field);
  READ_PARAMETER(dict, ion_charge_state);
  READ_PARAMETER(dict, energy_min);
  READ_PARAMETER(dict, energy_max);
  READ_PARAMETER(dict, energy_num);
  READ_PARAMETER(dict, phix_min);
  READ_PARAMETER(dict, phix_max);
  READ_PARAMETER(dict, phix_num);
  READ_PARAMETER(dict, phiy_min);
  READ_PARAMETER(dict, phiy_max);
  READ_PARAMETER(dict, phiy_num);
  READ_PARAMETER(dict, energy_scale);
  assert(energy_scale == "linear" || energy_scale == "log");
  try {
    std::string seed_str;
    get_from_dict(dict, "seed", seed_str);
    seed = boost::numeric_cast<unsigned>(std::stoull(seed_str));
  } catch (std::runtime_error&) {
    if (world.rank() == 0) {
      seed = std::time(nullptr);
      for (int i = 1; i != world.size(); ++i) {
        world.send(i, 0, &seed, 1);
      }
    } else {
      world.recv(0, 0, &seed, 1);
    }
  }
  READ_PARAMETER(dict, steps);
  READ_PARAMETER(dict, particles_target);
  READ_PARAMETER(dict, solver);
  assert(solver == "euler" || solver == "rk4");
  READ_PARAMETER(dict, output_trajectories);
  READ_PARAMETER(dict, general_output_filename);
  READ_PARAMETER(dict, radiation_output_filename);
  READ_PARAMETER(dict, particles_output_filename);

  if (world.rank() == 0) {
    for (auto pair : dict) {
      std::cerr << "WARNING: extra parameter '" << pair.first <<
        "' defined in input file with value '" << pair.second << "'" <<
        std::endl;
    }
  }

  computeDependentParameters(world.size());
}


void Parameters::computeDependentParameters(int processes)
{
  plasma_frequency = elementary_charge * std::sqrt(plasma_density / (vacuum_permittivity * electron_mass));
  plasma_angular_wavenumber = plasma_frequency / c_light;
  plasma_skin_depth = 1 / plasma_angular_wavenumber;
  gamma_initial = beam_energy_initial_ev / electron_rest_energy_ev;
  time_step = plasma_length / (c_light * (steps - 1));
  z_step = c_light * time_step;
  compute_processes = static_cast<std::size_t>(processes) - 1;
  particles_per_process = particles_target / compute_processes;
  actual_particles = particles_per_process * compute_processes;
  assert(actual_particles % compute_processes == 0);
}


#define WRITE_PARAMETER(file, param, units) file << #param " = " << param << " " units "\n"
#define WRITE_PARAMETER_BOOL(file, param, units) file << #param " = " << (param ? "true" : "false") << " " units "\n"

void Parameters::writeOutputFile(double seconds)
{
  try {
    std::ofstream file;
    file.exceptions(file.failbit | file.badbit);
    file.open(general_output_filename);
    WRITE_PARAMETER(file, mu_x, "m");
    WRITE_PARAMETER(file, mu_y, "m");
    WRITE_PARAMETER(file, sigma_x, "m");
    WRITE_PARAMETER(file, sigma_y, "m");
    WRITE_PARAMETER(file, emit_n_x, "m");
    WRITE_PARAMETER(file, emit_n_y, "m");
    WRITE_PARAMETER(file, beam_energy_initial_ev, "eV");
    WRITE_PARAMETER(file, percent_energy_spread, "%");
    WRITE_PARAMETER(file, plasma_length, "m");
    WRITE_PARAMETER(file, plasma_density, "m^-3");
    WRITE_PARAMETER(file, accelerating_field, "V/m");
    WRITE_PARAMETER(file, ion_charge_state, "-");
    WRITE_PARAMETER(file, energy_min, "eV");
    WRITE_PARAMETER(file, energy_max, "eV");
    WRITE_PARAMETER(file, energy_num, "-");
    WRITE_PARAMETER(file, phix_min, "rad");
    WRITE_PARAMETER(file, phix_max, "rad");
    WRITE_PARAMETER(file, phix_num, "-");
    WRITE_PARAMETER(file, phiy_min, "rad");
    WRITE_PARAMETER(file, phiy_max, "rad");
    WRITE_PARAMETER(file, phiy_num, "-");
    WRITE_PARAMETER(file, energy_scale, "-");
    WRITE_PARAMETER(file, seed, "-");
    WRITE_PARAMETER(file, steps, "-");
    WRITE_PARAMETER(file, particles_target, "-");
    WRITE_PARAMETER(file, solver, "-");
    WRITE_PARAMETER_BOOL(file, output_trajectories, "-");
    WRITE_PARAMETER(file, general_output_filename, "-");
    WRITE_PARAMETER(file, radiation_output_filename, "-");
    WRITE_PARAMETER(file, particles_output_filename, "-");
    WRITE_PARAMETER(file, plasma_frequency, "s^-1");
    WRITE_PARAMETER(file, plasma_angular_wavenumber, "m^-1");
    WRITE_PARAMETER(file, plasma_skin_depth, "m");
    WRITE_PARAMETER(file, gamma_initial, "-");
    WRITE_PARAMETER(file, time_step, "s");
    WRITE_PARAMETER(file, z_step, "m");
    WRITE_PARAMETER(file, compute_processes, "-");
    WRITE_PARAMETER(file, actual_particles, "-");
    WRITE_PARAMETER(file, particles_per_process, "-");
    file << "seconds_elapsed = " << seconds << " s\n";
    file << "minutes_elapsed = " << (seconds / 60) << " min\n";
    file << "hours_elapsed = " << (seconds / (60 * 60)) << " hours\n";
    file << "approx_core_hours_elapsed = " << ((compute_processes * seconds) /
      (60 * 60)) << " core-hours\n";
    file.close();
  } catch (const std::ifstream::failure&) {
    throw std::runtime_error("unable to write output file");
  }
}


struct Particle {
  double x, y, zeta, bx, by, g, bxd, byd, gd;
};

template <unsigned long n>
static std::array<double, n> arr_add(std::array<double, n> const& a, std::array<double, n> const& b)
{
  std::array<double, n> c;
  for (std::size_t i = 0; i != n; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <unsigned long n>
static std::array<double, n> arr_mul(double a, std::array<double, n> const& b)
{
  std::array<double, n> c;
  for (std::size_t i = 0; i != n; ++i) {
    c[i] = a * b[i];
  }
  return c;
}

static std::array<double, 3> cross(std::array<double, 3> a, std::array<double, 3> b)
{
  return {
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  };
}

static std::array<double, 3> subtract(std::array<double, 3> a, std::array<double, 3> b)
{
  return {
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2]
  };
}

static double dot(std::array<double, 3> a, std::array<double, 3> b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void runMainProcess(boost::mpi::communicator& world, const Parameters p)
{
  std::size_t grid_size = 6 * p.energy_num * p.phix_num * p.phiy_num;
  auto grid = std::make_unique<double[]>(grid_size);
  {
    auto temporary_grid = std::make_unique<double[]>(grid_size);
    for (int i = 0; i != world.size() - 1; ++i) {
      world.recv(i + 1, 1, reinterpret_cast<char*>(temporary_grid.get()),
        sizeof(double) * grid_size);
      std::cout << "received grid from process " << i + 1 << std::endl;
      for (std::size_t j = 0; j != grid_size; ++j)
        grid[j] += temporary_grid[j];
    }
  }
  std::cout << "writing data to file" << std::endl;
  std::ofstream output_file;
  output_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  output_file.open(p.radiation_output_filename, std::ofstream::binary);
  output_file.write(reinterpret_cast<char*>(grid.get()), sizeof(double) * grid_size);
  output_file.close();
}

static void runComputeProcess(boost::mpi::communicator& world, const Parameters p)
{
  // Create a result grid
  auto grid = std::make_unique<double[]>(6 * p.energy_num * p.phix_num * p.phiy_num);
  for (std::size_t i = 0; i != 6 * p.energy_num * p.phix_num * p.phiy_num; ++i)
    grid[i] = 0;

  // Compute step sizes
  double energy_step;
  if (p.energy_scale == "log") {
    energy_step = p.energy_num == 1 ? 0 : (std::log(p.energy_max) - std::log(p.energy_min)) / (p.energy_num - 1);
  }
  else {
    energy_step = p.energy_num == 1 ? 0 : (p.energy_max - p.energy_min) / (p.energy_num - 1);
  }
  double phix_step = p.phix_num == 1 ? 0 : (p.phix_max - p.phix_min) / (p.phix_num - 1);
  double phiy_step = p.phiy_num == 1 ? 0 : (p.phiy_max - p.phiy_min) / (p.phiy_num - 1);

  // Setup random
  boost::random::mt19937 twister;
  if (p.seed == 0) {
    twister.seed(twister() + world.rank());
  } else {
    twister.seed(p.seed + world.rank());
  }
  boost::random::normal_distribution x_dist{p.mu_x, p.sigma_x};
  boost::random::normal_distribution y_dist{p.mu_y, p.sigma_y};
  boost::random::normal_distribution gbx_dist{0.0, p.emit_n_x / p.sigma_x};
  boost::random::normal_distribution gby_dist{0.0, p.emit_n_y / p.sigma_y};

  // Output file
  std::ofstream output_file;
  if (p.output_trajectories) {
    output_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    output_file.open(p.particles_output_filename + "_" + std::to_string(world.rank()), std::ofstream::binary);
  }

  // For each particle
  for (std::size_t i = 0; i != p.particles_per_process; ++i) {
    std::cout << "particle " << i + 1 << " of " << p.particles_per_process << std::endl;
    std::array<double, 5> coordinates;
    coordinates[0] = x_dist(twister);
    coordinates[1] = y_dist(twister);
    coordinates[2] = 0;
    coordinates[3] = gbx_dist(twister);
    coordinates[4] = gby_dist(twister);

    double a = (1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2)) * std::pow(p.gamma_initial, -2);
    double constant_1 = p.ion_charge_state * std::pow(elementary_charge, 2) * p.plasma_density / (2 * vacuum_permittivity * electron_mass * c_light);
    double constant_2 = std::sqrt(std::pow(p.gamma_initial, 2) - std::pow(coordinates[3], 2) - std::pow(coordinates[4], 2) - 1);
    double constant_3 = -elementary_charge * p.accelerating_field / (electron_mass * c_light);
    double constant_4 = p.gamma_initial * (-0.5 * a - 0.125 * a * a - 0.0625 * a * a * a * a);

    double f_gamma_initial = p.gamma_initial;
    std::function<std::array<double, 5>(double, std::array<double, 5>)> f = [constant_1, constant_2, constant_3, constant_4, f_gamma_initial](double t, std::array<double, 5> coordinates){
     double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_2 + constant_3 * t, 2));
     std::array<double, 5> rhs;
     rhs[0] = coordinates[3] * c_light / gamma;
     rhs[1] = coordinates[4] * c_light / gamma;
     rhs[2] = c_light * (constant_4 + constant_3 * t + (f_gamma_initial - gamma)) / gamma;
     rhs[3] = -constant_1 * coordinates[0];
     rhs[4] = -constant_1 * coordinates[1];
     return rhs;
    };

    // For each step
    std::size_t step = 0;
    while (true) {
      double t = step * p.time_step;

      // Convert
      Particle particle;
      {
        double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_2 + constant_3 * t, 2));
        double bx = coordinates[3] / gamma;
        double by = coordinates[4] / gamma;
        double value = -constant_1;
        double gamma_dot = (constant_3 * (constant_2 + constant_3 * t) + value * (coordinates[0] * coordinates[3] + coordinates[1] * coordinates[4])) / gamma;
        double bxd = (value * coordinates[0] / gamma) - (coordinates[3] * gamma_dot * std::pow(gamma, -2));
        double byd = (value * coordinates[1] / gamma) - (coordinates[4] * gamma_dot * std::pow(gamma, -2));
        particle.x = coordinates[0];
        particle.y = coordinates[1];
        particle.zeta = coordinates[2];
        particle.bx = bx;
        particle.by = by;
        particle.g = gamma;
        particle.bxd = bxd;
        particle.byd = byd;
        particle.gd = gamma_dot;
      }

      if (p.output_trajectories) {
        output_file.write(reinterpret_cast<char*>(&particle), sizeof(Particle));
      }

      // For each grid point
      for (std::size_t energy_idx = 0; energy_idx != p.energy_num; ++energy_idx) {
        double frequency;
        if (p.energy_scale == "log") {
          frequency = std::exp(std::log(p.energy_min) + energy_idx * energy_step) / hbar_ev;
        }
        else {
          frequency = (p.energy_min + energy_idx * energy_step) / hbar_ev;
        }
        for (std::size_t phix_idx = 0; phix_idx != p.phix_num; ++phix_idx) {
          double phix = p.phix_min + phix_idx * phix_step;
          for (std::size_t phiy_idx = 0; phiy_idx != p.phiy_num; ++phiy_idx) {
            double phiy = p.phiy_min + phiy_idx * phiy_step;
            std::size_t result_index = 6 * (phiy_idx + p.phiy_num * (phix_idx + p.phix_num * (energy_idx)));
            // Compute radiation
            std::array<double, 3> n;
            std::array<double, 3> b;
            std::array<double, 3> bd;
            n[0] = std::sin(phix);
            n[1] = std::sin(phiy);
            n[2] = std::sqrt(1 - std::pow(n[0], 2) - std::pow(n[1], 2));
            b[0] = particle.bx;
            b[1] = particle.by;
            b[2] = std::sqrt(1 - std::pow(particle.g, -2) - std::pow(particle.bx, 2) - std::pow(particle.by, 2));
            //constexpr double THETA = 5e-4;
            if (1 - dot(b, n) - 0.5 * std::pow(particle.g, -2) > 2 * std::pow(std::sin(0.5 * p.cutoff), 2))
              continue;
            bd[0] = particle.bxd;
            bd[1] = particle.byd;
            bd[2] = -(particle.gd * std::pow(particle.g, -3) + particle.bx * particle.bxd + particle.by * particle.byd) / b[2];
            auto vector = cross(n, cross(subtract(n, b), bd));
            auto denom = std::pow(1 - dot(b, n), -2);
            double n_transverse2 = std::pow(n[0], 2) + std::pow(n[1], 2);
            double value = 0.5 * n_transverse2 + 0.125 * std::pow(n_transverse2, 2) + 0.0625 * std::pow(n_transverse2, 3);
            double phase = frequency * ((t * value) - (n[0] * particle.x + n[1] * particle.y + n[2] * particle.zeta) / c_light);
            double exponential_real = std::cos(phase);
            double exponential_imag = std::sin(phase);
            double integration_multiplier = (step == 0 || step == (p.steps - 1)) ? 0.5 : 1.0;

            grid[result_index + 0] += vector[0] * exponential_real * denom * integration_multiplier * p.time_step * constant;
            grid[result_index + 1] += vector[0] * exponential_imag * denom * integration_multiplier * p.time_step * constant;
            grid[result_index + 2] += vector[1] * exponential_real * denom * integration_multiplier * p.time_step * constant;
            grid[result_index + 3] += vector[1] * exponential_imag * denom * integration_multiplier * p.time_step * constant;
            grid[result_index + 4] += vector[2] * exponential_real * denom * integration_multiplier * p.time_step * constant;
            grid[result_index + 5] += vector[2] * exponential_imag * denom * integration_multiplier * p.time_step * constant;
          }
        }
      }

      // Check to see if done
      if (step == p.steps - 1)
        break;

      // Compute next step
      if (p.solver == "euler") {
        coordinates = arr_add(coordinates, arr_mul(p.time_step, f(t, coordinates)));
      }
      else {
        std::array<double, 5> k1 = arr_mul(p.time_step, f(t, coordinates));
        std::array<double, 5> k2 = arr_mul(p.time_step, f(t + 0.5 * p.time_step, arr_add(coordinates, arr_mul(0.5, k1))));
        std::array<double, 5> k3 = arr_mul(p.time_step, f(t + 0.5 * p.time_step, arr_add(coordinates, arr_mul(0.5, k2))));
        std::array<double, 5> k4 = arr_mul(p.time_step, f(t + p.time_step, arr_add(coordinates, k3)));
        coordinates = arr_add(coordinates, arr_mul(1/6.0, arr_add(arr_add(k1, k4), arr_mul(2.0, arr_add(k2, k3)))));
      }
      ++step;
    }
  }

  world.send(0, 1, reinterpret_cast<char*>(grid.get()), sizeof(double) * 6 * p.energy_num * p.phix_num * p.phiy_num);
  if (p.output_trajectories) {
    output_file.close();
  }
}


int main(int argc, char* argv[])
{
  // Start Timer
  std::time_t begin;
  std::time(&begin);

  // Initialize io
  std::ios_base::sync_with_stdio(false);
  std::cin.exceptions(std::ios_base::badbit | std::ios_base::failbit);
  std::cout.exceptions(std::ios_base::badbit | std::ios_base::failbit);
  std::cerr.exceptions(std::ios_base::badbit | std::ios_base::failbit);
  std::clog.exceptions(std::ios_base::badbit | std::ios_base::failbit);

  // Initialize MPI
  boost::mpi::environment env;
  boost::mpi::communicator world;

  // Parameters
  if (argc != 2)
    throw std::runtime_error("expected a single argument containing the path to"
      "the input file");

  if (world.size() == 1)
     throw std::runtime_error("you need at least two processes to run this program");

  Parameters parameters{argv[1], world};

  // Run Program
  if (world.rank() == 0) {
    runMainProcess(world, parameters);
    std::time_t end;
    std::time(&end);
    double diff = std::difftime(end, begin);
    parameters.writeOutputFile(diff);
    return EXIT_SUCCESS;
  }
  else {
    runComputeProcess(world, parameters);
    return EXIT_SUCCESS;
  }
}
