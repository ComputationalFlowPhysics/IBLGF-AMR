#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <dictionary/dictionary.hpp>
#include "gtest/gtest.h"
#include "../vortexring/vortexring.hpp"

namespace {

// The fixture for testing class BasicTest
class BasicTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  BasicTest() {
     // You can do set-up work for each test here.
  }

  ~BasicTest() override {
     // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

  // Objects declared here can be used by all tests in the test suite for Foo.


};

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}

// The fixture for testing class BasicTest
class VortexRing : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  VortexRing() {
     // You can do set-up work for each test here.
  }

  ~VortexRing() override {
     // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

  // Objects declared here can be used by all tests in the test suite for Foo.
};

// Tests that the Foo::Bar() method does Abc.
TEST_F(BasicTest, EqualOne) {
  // Foo f;
  // EXPECT_EQ(f.Bar(input_filepath, output_filepath), 0);
  EXPECT_EQ(1, 1);
}

// Tests that Foo does Xyz.
TEST_F(BasicTest, EqualString) {
  // Exercises the Xyz feature of Foo.
  const std::string str1 = "this/package/testdata/test.dat";
  const std::string str2 = "this/package/testdata/test.dat";
  EXPECT_EQ(str1,str2);
}



TEST_F(VortexRing, OutputFileMatch) {
//	const std::string str2 = "this/package/testdata/test.dat";
//
	boost::mpi::communicator world;

	std::string input = "/home/mlee/IBLGF-AMR-parallel-reapply/tests/googletest/configFile";
	//std::string input = "./configFile";

    // Read in dictionary
//    Dictionary dictionary(input);

    //Instantiate setup
//    VortexRingTest setup(&dictionary);

    // run setup
//    setup.run();

	// Check output file
	// Binary files will not differ
	// std::string output = exec("diff mesh.hdf5 test-vortexRing.hdf5");
	// EXPECT_EQ(output,"");

	std::string output;
	// Check h5dump
	// Full
	exec("h5dump mesh.hdf5 > vortexRing.txt");
	output = exec("diff vortexRing.txt test-vortexRing.txt");
	EXPECT_EQ(output,"1c1\n< HDF5 \"mesh.hdf5\" {\n---\n> HDF5 \"test-vortexRing.hdf5\" {\n");

	// Header only
	exec("h5dump --header mesh.hdf5 > vortexRing-headers.txt");
	output = exec("diff vortexRing-headers.txt test-vortexRing-headers.txt");
	EXPECT_EQ(output,"1c1\n< HDF5 \"mesh.hdf5\" {\n---\n> HDF5 \"test-vortexRing.hdf5\" {\n");

	// Attributes
	exec("h5dump -A mesh.hdf5 > vortexRing-attrs.txt");
	output = exec("diff vortexRing-attrs.txt test-vortexRing-attrs.txt");
	EXPECT_EQ(output,"1c1\n< HDF5 \"mesh.hdf5\" {\n---\n> HDF5 \"test-vortexRing.hdf5\" {\n");

	// Dataset of different levels:
	exec("h5dump --dataset=/level_0/data:datatype=0 mesh.hdf5 > vortexRing-dset-lvl0.txt");
	output = exec("diff vortexRing-dset-lvl0.txt test-vortexRing-dset-lvl0.txt");
	EXPECT_EQ(output,"1c1\n< HDF5 \"mesh.hdf5\" {\n---\n> HDF5 \"test-vortexRing.hdf5\" {\n");

	exec("h5dump --dataset=/level_1/data:datatype=0 mesh.hdf5 > vortexRing-dset-lvl1.txt");
	output = exec("diff vortexRing-dset-lvl1.txt test-vortexRing-dset-lvl1.txt");
	EXPECT_EQ(output,"1c1\n< HDF5 \"mesh.hdf5\" {\n---\n> HDF5 \"test-vortexRing.hdf5\" {\n");

	exec("h5dump --dataset=/level_2/data:datatype=0 mesh.hdf5 > vortexRing-dset-lvl2.txt");
	output = exec("diff vortexRing-dset-lvl2.txt test-vortexRing-dset-lvl2.txt");
	EXPECT_EQ(output,"1c1\n< HDF5 \"mesh.hdf5\" {\n---\n> HDF5 \"test-vortexRing.hdf5\" {\n");

	exec("h5dump --dataset=/level_3/data:datatype=0 mesh.hdf5 > vortexRing-dset-lvl3.txt");
	output = exec("diff vortexRing-dset-lvl3.txt test-vortexRing-dset-lvl3.txt");
	EXPECT_EQ(output,"1c1\n< HDF5 \"mesh.hdf5\" {\n---\n> HDF5 \"test-vortexRing.hdf5\" {\n");

	exec("h5dump --dataset=/level_4/data:datatype=0 mesh.hdf5 > vortexRing-dset-lvl4.txt");
	output = exec("diff vortexRing-dset-lvl4.txt test-vortexRing-dset-lvl4.txt");
	ASSERT_EQ(output,"1c1\n< HDF5 \"mesh.hdf5\" {\n---\n> HDF5 \"test-vortexRing.hdf5\" {\n");
	//
}



}  // namespace



int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc,argv);
    std::cout<<"Running tests"<<std::endl;

	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
//
//	std::string input="./";
//    input += std::string("configFile");
//
//    if (argc>1 && argv[1][0] != '-')
//    {
//        input = argv[1];
//    }
//
//    // Read in dictionary
//    Dictionary dictionary(input);
//
//    //Instantiate setup
//    VortexRingTest setup(&dictionary);
//
//    // run setup
//    setup.run();
//
//    return 0;
    return RUN_ALL_TESTS();
}
