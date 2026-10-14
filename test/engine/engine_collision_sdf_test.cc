// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests for sdf collisions.

#include "src/engine/engine_collision_sdf.h"

#include <cstdio>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using SdfTest = MujocoTest;

// Trilinear coefficients store distances in meters; derivatives with respect
// to unit voxel coordinates must include the inverse physical voxel width.
TEST_F(SdfTest, MeshGradientMatchesPhysicalFiniteDifference) {
  constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="box" inertia="exact"
        vertex="-.01 -.02 -.04 .01 -.02 -.04 .01 .02 -.04 -.01 .02 -.04
                -.01 -.02 .04 .01 -.02 .04 .01 .02 .04 -.01 .02 .04"
        face="3 7 6 3 6 2 0 1 5 0 5 4 0 3 2 0 2 1
              4 5 6 4 6 7 0 4 7 0 7 3 1 2 6 1 6 5"/>
    </asset>
    <worldbody><body><geom type="sdf" mesh="box"/></body></worldbody>
  </mujoco>)";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int id = 0;
  const mjpPlugin* plugin = nullptr;
  mjtGeom type = mjGEOM_SDF;
  mjSDF sdf{};
  sdf.id = &id;
  sdf.plugin = &plugin;
  sdf.type = mjSDFTYPE_SINGLE;
  sdf.geomtype = &type;
  ASSERT_GE(model->mesh_octadr[0], 0);
  const mjtNum* box = model->oct_aabb + 6 * model->mesh_octadr[0];
  for (int face_axis = 0; face_axis < 3; ++face_axis) {
    mjtNum point[3], gradient[3];
    for (int axis = 0; axis < 3; ++axis) {
      point[axis] =
          box[axis] + (axis == face_axis ? .873 : .123) * box[3 + axis];
    }
    mjc_gradient(model.get(), data.get(), &sdf, gradient, point);
    for (int axis = 0; axis < 3; ++axis) {
      mjtNum plus[3], minus[3];
      mju_copy3(plus, point);
      mju_copy3(minus, point);
      plus[axis] += 1e-7;
      minus[axis] -= 1e-7;
      mjtNum difference = (mjc_distance(model.get(), data.get(), &sdf, plus) -
                           mjc_distance(model.get(), data.get(), &sdf, minus)) /
                          2e-7;
      EXPECT_NEAR(gradient[axis], difference, MjTol(1e-6, 1e-2));
    }
  }
}

static constexpr int kpoints = 6;
static constexpr int kgeoms = 5;
static constexpr char kSdfModel[] = R"(
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 .1" pos="0 0 -1"/>
    <body pos="-.1 .2 2" euler="0 0 45">
      <geom type="sphere" size="1"/>
      <geom type="capsule" size=".1" fromto="-2 -2 1 2 2 1"/>
      <geom type="cylinder" size="1" fromto="0 0 0 2 0 0"/>
      <geom type="box" size="1 1 1"/>
    </body>
  </worldbody>
</mujoco>
)";

TEST_F(SdfTest, SdfPrimitive) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(kSdfModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());
  ASSERT_THAT(model->ngeom, kgeoms);

  mjSDF sdf;
  const mjpPlugin* null_plugin = NULL;
  mjtNum gradient[3],
      dist[kgeoms][kpoints] = {
          {0, 0, 0, 0, 1, 1},  // plane
          {-1, 0, 0, mju_sqrt(2) - 1, mju_sqrt(2) - 1,
           mju_sqrt(3) - 1},  // sphere
          {(mjtNum)-.1, (mjtNum).9, (mjtNum).9, mju_sqrt(2) - (mjtNum).1,
           (mjtNum).9, mju_sqrt(2) - (mjtNum).1},           // capsule
          {-1, 0, 0, mju_sqrt(2) - 1, 0, mju_sqrt(2) - 1},  // cylinder
          {-mju_sqrt(3), 0, 0, 0, 0, 0},                    // box
      };
  mjtNum points[kpoints][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                               {1, 1, 0}, {0, 1, 1}, {1, 1, 1}};

  for (int i = 0; i < kgeoms; i++) {
    sdf.plugin = &null_plugin;
    sdf.id = &i;
    sdf.type = mjSDFTYPE_SINGLE;
    sdf.geomtype = (mjtGeom*)(model->geom_type + i);
    for (int j = 0; j < kpoints; j++) {
      EXPECT_NEAR(mjc_distance(model.get(), data.get(), &sdf, points[j]),
                  dist[i][j], MjTol(1e-9, 5e-7));
      mjc_gradient(model.get(), data.get(), &sdf, gradient, points[j]);
    }
  }
}

static constexpr char kFlexSdfModel[] = R"(
<mujoco>
  <extension>
    <plugin plugin="mujoco.sdf.torus">
      <instance name="torus">
        <config key="radius1" value="0.35"/>
        <config key="radius2" value="0.15"/>
      </instance>
    </plugin>
  </extension>

  <asset>
    <mesh name="torus">
      <plugin instance="torus"/>
    </mesh>
  </asset>

  <worldbody>
    <body pos="0 0 -0.3">
      <geom type="sdf" mesh="torus">
        <plugin instance="torus"/>
      </geom>
    </body>
    <body name="flex">
      <flexcomp name="test" type="grid" count="3 3 1" spacing=".2 .2 .2" dim="2">
        <elasticity young="1e4"/>
        <contact selfcollide="none" internal="false"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
)";

TEST_F(SdfTest, FlexSdfCollision) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(kFlexSdfModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  // check there is at least one flex
  ASSERT_GT(model->nflex, 0);

  // simulate for a few steps to let flex fall onto torus
  for (int i = 0; i < 100; i++) {
    mj_step(model.get(), data.get());
  }

  // should have contacts between flex and SDF
  EXPECT_GT(data->ncon, 0) << "Expected flex-SDF contacts";

  // check that contacts involve the flex
  bool has_flex_contact = false;
  for (int i = 0; i < data->ncon; i++) {
    if (data->contact[i].flex[0] >= 0 || data->contact[i].flex[1] >= 0) {
      has_flex_contact = true;
      break;
    }
  }
  EXPECT_TRUE(has_flex_contact) << "Expected at least one flex contact";
}

}  // namespace
}  // namespace mujoco
