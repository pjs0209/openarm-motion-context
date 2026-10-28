// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  CSS2DObject,
  CSS2DRenderer,
} from "three/addons/renderers/CSS2DRenderer.js";
import URDFLoader from "/vendor/urdf-loader/src/URDFLoader.js";

const $ = (id) => document.getElementById(id);
const names = ["box", "ball", "peg", "hole"];
const colors = {
  box: "#e7ae59",
  ball: "#f2737d",
  peg: "#6cafff",
  hole: "#6de0bb",
};
const layers = {
  measured: true,
  target: true,
  box: true,
  ball: true,
  peg: true,
  hole: true,
  axes: true,
  grid: true,
  reference: true,
};
const viewport = $("viewport");
const scene = new THREE.Scene();
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(1.65, -1.8, 1.35);
const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: true,
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.prepend(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.className = "labels-layer";
viewport.append(labelRenderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.15, 0, 0.57);
controls.enableDamping = true;
controls.dampingFactor = 0.09;
controls.minDistance = 0.35;
controls.maxDistance = 6;
controls.maxPolarAngle = Math.PI * 0.94;
scene.add(new THREE.HemisphereLight(0xbdddf4, 0x233843, 1.25));
const keyLight = new THREE.DirectionalLight(0xe2f4ff, 2.0);
keyLight.position.set(1.5, -2, 4);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
Object.assign(keyLight.shadow.camera, {
  left: -2,
  right: 2,
  top: 2,
  bottom: -2,
  near: 0.1,
  far: 10,
});
keyLight.shadow.bias = -0.0003;
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0x73bda8, 0.7);
fillLight.position.set(-2, 1, 1.8);
scene.add(fillLight);
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(8, 8),
  new THREE.ShadowMaterial({ opacity: 0.22 }),
);
ground.position.z = -0.008;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(5, 50, 0x3c675f, 0x28414e);
grid.rotation.x = Math.PI / 2;
grid.material.transparent = true;
grid.material.opacity = 0.38;
scene.add(grid);
const originAxes = new THREE.AxesHelper(0.13);
scene.add(originAxes);
const objectsRoot = new THREE.Group();
const graspRoot = new THREE.Group();
const objectFrame = new THREE.Group();
objectFrame.matrixAutoUpdate = false;
objectFrame.add(objectsRoot, graspRoot);
scene.add(objectFrame);
const referenceRoot = new THREE.Group();
scene.add(referenceRoot);
const referenceGroups = {};
let usdScene = null;
const objectGroups = {};
const graspGroups = {};
const arrows = {};
const robotAxes = [];
let measuredRobot = null;
let targetRobot = null;
let latest = null;
let connected = false;
let modelReady = false;
let lastPoll = performance.now();
let measuredOpacity = 1;
let initialViewFitted = false;

for (const name of names) {
  const card = document.createElement("div");
  card.className = "object-card";
  card.innerHTML = `<div class="object-card-top"><i class="swatch ${name}"></i><strong>${name[0].toUpperCase() + name.slice(1)}</strong><span class="data-status">NO DATA</span></div><div class="position-row">${["x", "y", "z"].map((axis) => `<div><span class="coordinate">${axis.toUpperCase()}</span><span data-position="${axis}">—</span></div>`).join("")}</div><div class="object-age">No pose received</div>`;
  card.dataset.object = name;
  $("object-cards").append(card);
}
for (const side of ["left", "right"]) {
  const table = document.createElement("div");
  table.className = "arm-table";
  table.dataset.arm = side;
  table.innerHTML = `<h3><i class="swatch ${side === "left" ? "target" : "measured"}"></i>${side[0].toUpperCase() + side.slice(1)} arm<span>7 JOINTS</span></h3><table><thead><tr><th>JOINT</th><th>MEASURED</th><th>TARGET</th><th>ABS. ERROR</th></tr></thead><tbody>${Array.from({ length: 7 }, (_, i) => `<tr><td>J${i + 1}</td><td>—</td><td>—</td><td>—</td></tr>`).join("")}</tbody></table>`;
  $("arm-tables").append(table);
}

function setMessage(text) {
  $("scene-message").textContent = text;
  $("scene-message").hidden = !text;
}

function resize() {
  const { width, height } = viewport.getBoundingClientRect();
  renderer.setSize(width, height);
  labelRenderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

function tintRobot(robot, ghost) {
  for (const joint of Object.values(robot.joints)) joint.ignoreLimits = true;
  robot.traverse((object) => {
    if (!object.isMesh) return;
    object.castShadow = !ghost;
    object.receiveShadow = !ghost;
    const convert = () => {
      const material = new THREE.MeshStandardMaterial();
      material.color?.set(ghost ? 0x72e3ca : 0xa8bfcd);
      material.transparent = ghost;
      material.opacity = ghost ? 0.25 : 1;
      material.depthWrite = !ghost;
      if ("roughness" in material) material.roughness = 0.65;
      if ("metalness" in material) material.metalness = 0.2;
      return material;
    };
    object.material = Array.isArray(object.material)
      ? object.material.map(convert)
      : convert(object.material);
    object.renderOrder = ghost ? 2 : 0;
  });
}

async function loadRobot() {
  try {
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    loader.packages = { openarm_description: "/assets/openarm_description" };
    await new Promise((resolve, reject) => {
      manager.onLoad = resolve;
      manager.onError = (url) =>
        reject(new Error(`Robot asset failed: ${url}`));
      loader.load(
        "/api/robot.urdf",
        (robot) => {
          measuredRobot = robot;
        },
        undefined,
        reject,
      );
    });
    if (!measuredRobot) throw new Error("Robot description did not load");
    targetRobot = measuredRobot.clone(true);
    tintRobot(measuredRobot, false);
    tintRobot(targetRobot, true);
    measuredRobot.visible = false;
    targetRobot.visible = false;
    scene.add(measuredRobot, targetRobot);
    const base =
      measuredRobot.links[latest?.base_frame || "openarm_body_link0"];
    if (!base)
      throw new Error(
        "The configured robot base frame is absent from the URDF",
      );
    measuredRobot.updateMatrixWorld(true);
    objectFrame.matrix.copy(base.matrixWorld);
    for (const side of ["left", "right"]) {
      for (const robot of [measuredRobot, targetRobot]) {
        const hand = robot.links[`openarm_${side}_hand_tcp`];
        if (hand) {
          const axes = new THREE.AxesHelper(0.075);
          hand.add(axes);
          robotAxes.push(axes);
        }
      }
      const arrow = new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(),
        0.1,
        0x8ce0c8,
        0.02,
        0.01,
      );
      arrow.visible = false;
      scene.add(arrow);
      arrows[side] = arrow;
    }
    modelReady = true;
    $("model-state").textContent = "V1 CAD · loaded";
    $("model-state").classList.add("success-text");
    if (latest) update(latest);
    else setMessage("Waiting for robot joint states…");
  } catch (error) {
    $("model-state").textContent = "Robot model unavailable";
    $("model-state").classList.add("error-text");
    setMessage(error.message);
    console.error(error);
  }
}

async function loadUsdScene() {
  try {
    const response = await fetch("/api/usd-scene", { cache: "no-store" });
    if (!response.ok) throw new Error(`USD scene HTTP ${response.status}`);
    usdScene = await response.json();
    if (!usdScene) return;
    $("usd-source").textContent =
      `${usdScene.source} · ${usdScene.frame} · STATIC REFERENCE`;
    for (const [name, object] of Object.entries(usdScene.objects)) {
      if (!names.includes(name)) continue;
      const group = objectMesh(name, { size: [0.1, 0.1, 0.1] }, true);
      referenceRoot.add(group);
      referenceGroups[name] = group;
      poseGroup(group, object.pose);
      const row = document.createElement("div");
      row.textContent = `${name}: ${object.pose.position.map((v) => v.toFixed(3)).join(" / ")} m`;
      row.title = object.prim;
      $("usd-positions").append(row);
      // A scene can arrive after the first ROS snapshot and proxy creation.
      const old = objectGroups[name];
      if (old) {
        old.removeFromParent();
        old.traverse((part) => {
          part.geometry?.dispose();
          if (part.material) part.material.dispose();
          if (part.isCSS2DObject) part.element.remove();
        });
        delete objectGroups[name];
      }
    }
    applyLayers();
    if (latest) update(latest);
  } catch (error) {
    $("usd-source").textContent = `USD unavailable: ${error.message}`;
    console.error(error);
  }
}

function objectMesh(name, data, reference = false) {
  const size = [...data.size];
  let geometry;
  if (data.shape === "sphere")
    geometry = new THREE.SphereGeometry(size[0] / 2, 32, 20);
  else if (data.shape === "cylinder") {
    geometry = new THREE.CylinderGeometry(
      size[0] / 2,
      size[1] / 2,
      size[2],
      40,
    );
    geometry.rotateX(Math.PI / 2);
  } else if (data.shape === "ring")
    geometry = new THREE.TorusGeometry(
      size[0] / 2,
      Math.min(size[0] * 0.12, 0.005),
      12,
      48,
    );
  else geometry = new THREE.BoxGeometry(...size);
  const group = new THREE.Group();
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color: colors[name],
      metalness: 0.22,
      roughness: 0.5,
      transparent: name === "box",
      opacity: name === "box" ? 0.48 : 0.95,
    }),
  );
  mesh.castShadow = true;
  const usd = usdScene?.objects[name];
  if (usd) {
    geometry.dispose();
    mesh.material.dispose();
    for (const shape of usd.shapes) {
      let actual;
      if (shape.type === "mesh") {
        actual = new THREE.BufferGeometry();
        actual.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(shape.positions, 3),
        );
        actual.setIndex(shape.indices);
        actual.computeVertexNormals();
      } else if (shape.type === "sphere") {
        actual = new THREE.SphereGeometry(shape.radius, 32, 20);
        actual.applyMatrix4(new THREE.Matrix4().fromArray(shape.matrix));
      } else throw new Error(`Unsupported USD shape ${shape.type}`);
      const part = new THREE.Mesh(
        actual,
        new THREE.MeshStandardMaterial({
          color: colors[name],
          roughness: 0.65,
          metalness: 0.1,
          side: THREE.DoubleSide,
          transparent: reference,
          opacity: reference ? 0.3 : 0.9,
          depthWrite: !reference,
        }),
      );
      part.castShadow = !reference;
      group.add(part);
    }
    const bounds = new THREE.Box3().setFromObject(group);
    size[2] = bounds.max.z * 2;
  } else group.add(mesh);
  if (!usd) {
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, 30),
      new THREE.LineBasicMaterial({
        color: colors[name],
        transparent: true,
        opacity: 0.75,
      }),
    );
    group.add(edges);
  }
  const label = document.createElement("div");
  label.className = "object-label";
  label.style.setProperty("--object-color", colors[name]);
  label.textContent =
    name[0].toUpperCase() +
    name.slice(1) +
    (reference ? " · USD reference" : "");
  const labelObject = new CSS2DObject(label);
  labelObject.position.z = size[2] / 2 + 0.045;
  group.add(labelObject);
  objectsRoot.add(group);
  return group;
}

function poseGroup(group, pose) {
  group.position.fromArray(pose.position);
  group.quaternion.fromArray(pose.quaternion);
}

function setMeasuredOpacity(opacity) {
  if (!measuredRobot || opacity === measuredOpacity) return;
  measuredRobot.traverse((object) => {
    if (!object.isMesh) return;
    for (const material of Array.isArray(object.material)
      ? object.material
      : [object.material]) {
      material.transparent = opacity < 1;
      material.opacity = opacity;
      material.depthWrite = opacity === 1;
    }
  });
  measuredOpacity = opacity;
}

function applyLayers() {
  grid.visible = layers.grid;
  referenceRoot.visible = layers.reference;
  for (const [name, group] of Object.entries(referenceGroups))
    group.visible = layers[name];
  originAxes.visible = layers.axes;
  for (const axes of robotAxes) axes.visible = layers.axes;
  const measured = latest?.robot?.measured;
  if (measuredRobot)
    measuredRobot.visible =
      layers.measured && Boolean(measured && Object.keys(measured).length);
  if (targetRobot)
    targetRobot.visible =
      layers.target &&
      connected &&
      Boolean(latest?.robot?.target_source) &&
      latest?.robot?.live;
  for (const [name, group] of Object.entries(objectGroups))
    group.visible =
      layers[name] && connected && latest?.objects?.[name]?.state === "LIVE";
  for (const group of Object.values(graspGroups))
    group.visible = layers.axes && connected && group.userData.live;
  for (const arrow of Object.values(arrows))
    arrow.visible =
      layers.target && layers.measured && connected && arrow.userData.live;
}

function updateTables(data) {
  for (const side of ["left", "right"]) {
    const rows = document.querySelectorAll(`[data-arm="${side}"] tbody tr`);
    (data.joint_order[`${side}_arm`] || []).forEach((name, index) => {
      const row = rows[index];
      if (!row) return;
      const current = data.robot.measured[name];
      const target = data.robot.target[name];
      row.cells[1].textContent = Number.isFinite(current)
        ? current.toFixed(3)
        : "—";
      row.cells[2].textContent = Number.isFinite(target)
        ? target.toFixed(3)
        : "—";
      const difference =
        Number.isFinite(target) && Number.isFinite(current)
          ? Math.abs(target - current)
          : null;
      row.cells[3].replaceChildren();
      row.cells[3].append(difference === null ? "—" : difference.toFixed(3));
      if (difference !== null) {
        const meter = document.createElement("span");
        meter.className = "error-meter";
        const fill = document.createElement("i");
        fill.style.width = `${Math.min(40, (difference / 0.15) * 40)}px`;
        if (difference > 0.08) fill.style.background = "#e7ae59";
        meter.append(fill);
        row.cells[3].append(meter);
      }
    });
  }
}

function updateCameras(data, demo) {
  const container = $("camera-cards");
  for (const [name, entry] of Object.entries(data.cameras)) {
    let row = [...container.children].find(
      (element) => element.dataset.camera === name,
    );
    if (!row) {
      row = document.createElement("div");
      row.className = "camera-row";
      row.dataset.camera = name;
      row.innerHTML =
        '<span class="camera-icon">▣</span><div class="camera-info"><span></span><span></span></div><span class="data-status"></span>';
      row.querySelector(".camera-info span").textContent = name
        .replaceAll("_", " ")
        .replace(/\b\w/g, (value) => value.toUpperCase());
      container.append(row);
    }
    row.querySelector(".camera-info span:last-child").textContent = entry.live
      ? `${entry.tags} tags · ${(entry.age_s * 1000).toFixed(0)} ms`
      : "No fresh detection array";
    const status = row.querySelector(".data-status");
    status.textContent = demo ? "DEMO" : entry.live ? "LIVE" : "NO DATA";
    status.className = `data-status${entry.live ? " live" : ""}`;
  }
}

function update(data) {
  if (!data.ready) return;
  latest = data;
  document.body.classList.remove("stream-offline");
  const demo = data.source === "demo";
  connected = true;
  $("connection-dot").className = "online";
  $("connection-text").textContent = demo ? "Local preview" : "Connected";
  $("source-badge").textContent = demo ? "DEMO" : "ROS LIVE";
  $("source-badge").className = demo ? "badge demo" : "badge neutral";
  $("demo-notice").hidden = !demo;
  $("source-detail").textContent = demo
    ? "SYNTHETIC DATA · NO ROS CONNECTION"
    : "LOCAL ROS TELEMETRY";
  $("mode-tag").textContent = demo
    ? "DEMO"
    : data.control.state.toUpperCase().replace("_", " ");
  $("mode-tag").className =
    `live-tag${data.control.state === "held" ? " held" : ""}`;
  $("robot-health").replaceChildren();
  $("robot-health").append(
    demo ? "Preview" : data.robot.live ? "Live" : "Stale",
  );
  const stateCaption = document.createElement("span");
  stateCaption.textContent = demo
    ? "synthetic scene"
    : data.robot.live
      ? "joint feedback"
      : "last known pose";
  $("robot-health").append(stateCaption);
  $("robot-age").textContent =
    data.robot.age_s === null
      ? "No measurement received"
      : `Observation age · ${(data.robot.age_s * 1000).toFixed(0)} ms`;
  const preview = data.robot.target_source === "preview";
  const targetName = preview
    ? "Dry-run candidate · not sent"
    : data.robot.target_source
      ? "Published final setpoint"
      : "No target received";
  $("target-caption").textContent = targetName;
  $("legend-target").textContent = preview
    ? "Dry-run candidate"
    : "Command target";
  $("joint-source").textContent = preview
    ? "PREVIEW"
    : data.robot.target_source
      ? "COMMAND"
      : "NO TARGET";
  $("error-label").firstChild.textContent = preview
    ? "PREVIEW OFFSET"
    : "TARGET OFFSET";
  $("base-frame").textContent = data.base_frame;
  $("object-frame").textContent = data.base_frame;
  if (usdScene) {
    const registered = usdScene.frame === data.base_frame;
    if (registered && referenceRoot.parent !== objectFrame)
      objectFrame.add(referenceRoot);
    else if (!registered && referenceRoot.parent !== scene)
      scene.add(referenceRoot);
    $("usd-source").textContent =
      `${usdScene.source} · ${usdScene.frame} · STATIC REFERENCE${registered ? "" : " · not registered to robot"}`;
  }
  $("control-hz").textContent = data.control_rate_hz;
  $("object-count").textContent = names.filter(
    (name) => data.objects[name]?.state === "LIVE",
  ).length;
  $("scene-update").textContent = demo
    ? "DEMO · SYNTHETIC"
    : data.robot.live
      ? "ROS · RECEIVING"
      : "ROS · WAITING";
  $("control-detail").textContent = data.control.detail || data.control.state;
  $("footer-dot").style.background =
    data.control.state === "held" ? "#e7ae59" : "#76dfc6";
  updateCameras(data, demo);
  updateTables(data);
  for (const name of names) {
    const object = data.objects[name];
    const card = document.querySelector(`[data-object="${name}"]`);
    const status = card.querySelector(".data-status");
    status.textContent = demo ? "DEMO" : object.state;
    status.className = `data-status${object.state === "LIVE" ? " live" : object.state === "STALE" ? " stale" : ""}`;
    for (const [index, axis] of ["x", "y", "z"].entries())
      card.querySelector(`[data-position="${axis}"]`).textContent = object.pose
        ? object.pose.position[index].toFixed(3)
        : "—";
    card.querySelector(".object-age").textContent =
      object.age_s === null
        ? "No pose received"
        : `${(object.age_s * 1000).toFixed(0)} ms observation age`;
  }
  if (modelReady) {
    const newBase = measuredRobot.links[data.base_frame];
    measuredRobot.setJointValues(data.robot.measured);
    targetRobot.setJointValues(data.robot.target);
    setMeasuredOpacity(data.robot.live ? 1 : 0.18);
    measuredRobot.updateMatrixWorld(true);
    targetRobot.updateMatrixWorld(true);
    if (newBase) {
      objectFrame.matrix.copy(newBase.matrixWorld);
      objectFrame.matrixWorldNeedsUpdate = true;
    }
    let maxDistance = 0;
    for (const side of ["left", "right"]) {
      const actual = measuredRobot.links[`openarm_${side}_hand_tcp`];
      const target = targetRobot.links[`openarm_${side}_hand_tcp`];
      const arrow = arrows[side];
      arrow.userData.live = false;
      if (actual && target && data.robot.target_source && data.robot.live) {
        const start = actual.getWorldPosition(new THREE.Vector3());
        const end = target.getWorldPosition(new THREE.Vector3());
        const direction = end.clone().sub(start);
        const distance = direction.length();
        maxDistance = Math.max(maxDistance, distance);
        if (distance > 0.0001) {
          arrow.position.copy(start);
          arrow.setDirection(direction.normalize());
          arrow.setLength(
            distance,
            Math.min(0.025, distance * 0.3),
            Math.min(0.012, distance * 0.15),
          );
          arrow.userData.live = true;
        }
      }
    }
    $("tcp-error").textContent =
      data.robot.target_source && data.robot.live
        ? (maxDistance * 1000).toFixed(1)
        : "—";
    for (const name of names) {
      const object = data.objects[name];
      if (object.pose) {
        objectGroups[name] ||= objectMesh(name, object);
        poseGroup(objectGroups[name], object.pose);
      }
    }
    for (const [side, grasp] of Object.entries(data.grasps)) {
      graspGroups[side] ||= new THREE.AxesHelper(0.06);
      const group = graspGroups[side];
      if (group.parent !== graspRoot) graspRoot.add(group);
      group.userData.live = grasp.state === "LIVE" && Boolean(grasp.pose);
      if (grasp.pose) poseGroup(group, grasp.pose);
    }
    if (!initialViewFitted && Object.keys(data.robot.measured).length) {
      initialViewFitted = true;
      $("fit-view").click();
    }
    if (Object.keys(data.robot.measured).length && data.robot.live)
      setMessage("");
    else
      setMessage(
        data.robot.age_s === null
          ? "Waiting for robot joint states…"
          : "Joint feedback is stale · showing the last known pose",
      );
    applyLayers();
  }
}

function disconnected(reason) {
  connected = false;
  document.body.classList.add("stream-offline");
  setMeasuredOpacity(0.18);
  $("connection-dot").className = "offline";
  $("connection-text").textContent = "Disconnected";
  $("mode-tag").textContent = "OFFLINE";
  $("mode-tag").className = "live-tag held";
  $("control-detail").textContent = reason;
  $("scene-update").textContent = "STREAM DISCONNECTED";
  $("robot-health").textContent = "Disconnected";
  $("robot-age").textContent = "Last pose is not live";
  $("tcp-error").textContent = "—";
  $("object-count").textContent = "0";
  $("joint-source").textContent = "STALE VALUES";
  for (const status of document.querySelectorAll(".data-status")) {
    status.textContent = "OFFLINE";
    status.className = "data-status stale";
  }
  for (const position of document.querySelectorAll("[data-position]"))
    position.textContent = "—";
  setMessage("Data stream unavailable · displayed robot pose is not live");
  for (const object of Object.values(objectGroups)) object.visible = false;
  applyLayers();
}

async function poll() {
  const start = performance.now();
  try {
    const response = await fetch("/api/state", {
      cache: "no-store",
      signal: AbortSignal.timeout(2000),
    });
    if (!response.ok)
      throw new Error(`Telemetry server returned ${response.status}`);
    const data = await response.json();
    if (data.snapshot_age_s > 0.6)
      throw new Error("The ROS observation stream has stopped updating");
    const now = performance.now();
    $("view-rate").textContent =
      `${(1000 / (now - lastPoll)).toFixed(1)} Hz view · ${(now - start).toFixed(0)} ms HTTP`;
    lastPoll = now;
    update(data);
  } catch (error) {
    disconnected(error.message);
  } finally {
    setTimeout(poll, 100);
  }
}

for (const input of document.querySelectorAll("[data-layer]"))
  input.addEventListener("change", () => {
    layers[input.dataset.layer] = input.checked;
    applyLayers();
  });
function setView(name) {
  const point = controls.target.clone();
  const distance = camera.position.distanceTo(point);
  if (name === "front")
    camera.position.copy(point).add(new THREE.Vector3(distance, 0, 0.04));
  else if (name === "top")
    camera.position.copy(point).add(new THREE.Vector3(0.001, 0, distance));
  else
    camera.position
      .copy(point)
      .add(
        new THREE.Vector3(1, -1.1, 0.6).normalize().multiplyScalar(distance),
      );
  controls.update();
  for (const button of document.querySelectorAll("[data-view]"))
    button.classList.toggle("selected", button.dataset.view === name);
}
for (const button of document.querySelectorAll("[data-view]"))
  button.addEventListener("click", () => setView(button.dataset.view));
$("fit-view").addEventListener("click", () => {
  if (!measuredRobot) return;
  const box = new THREE.Box3().setFromObject(measuredRobot);
  const objectBox = new THREE.Box3().setFromObject(objectsRoot);
  if (!objectBox.isEmpty()) box.union(objectBox);
  const referenceBox = new THREE.Box3().setFromObject(referenceRoot);
  if (!referenceBox.isEmpty()) box.union(referenceBox);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  if (!Number.isFinite(sphere.radius) || sphere.radius === 0) return;
  const direction = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(sphere.center);
  camera.position
    .copy(sphere.center)
    .add(
      direction.multiplyScalar(
        (sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) *
          1.1,
      ),
    );
  controls.update();
});
$("joint-nav").addEventListener("click", () =>
  $("joint-panel").scrollIntoView({ behavior: "smooth", block: "start" }),
);
$("scene-nav").addEventListener("click", () =>
  window.scrollTo({ top: 0, behavior: "smooth" }),
);
setInterval(() => {
  $("clock").textContent = new Date().toLocaleTimeString("en-GB", {
    hour12: false,
  });
}, 1000);
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}
resize();
animate();
loadRobot();
loadUsdScene();
poll();
