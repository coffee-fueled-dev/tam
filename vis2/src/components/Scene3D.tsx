import type { Frame, Metadata } from "@/types";
import { useEffect, useRef } from "react";
import * as THREE from "three";

interface Scene3DProps {
  frames: Frame[];
  metadata: Metadata;
  currentFrameIndex: number;
  maxHistory: number;
  reachedGoalsByFrame: Record<number, number[][]>;
}

// Colors matching the original HTML visualization
const COLORS = {
  trajectory: 0x6B7280,
  tube_high_agency: 0x2563EB,
  tube_low_agency: 0xCBD5E1,
  obstacle: 0xEF4444,
  goal: 0xF59E0B,
  goal_reached: 0xFBBF24,
  start: 0x1F2937,
  current_pos: 0x3B82F6, // Blue
  boundary: 0x9CA3AF,
};

export function Scene3D({
  frames,
  metadata,
  currentFrameIndex,
  maxHistory,
  reachedGoalsByFrame,
}: Scene3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const controlsRef = useRef<any>(null);
  const dynamicObjectsRef = useRef<THREE.Object3D[]>([]);
  const obstaclesRef = useRef<THREE.InstancedMesh | null>(null);
  const boundaryLinesRef = useRef<THREE.LineSegments | null>(null);
  const currentPosMarkerRef = useRef<THREE.Mesh | null>(null);
  const activeGoalMarkersRef = useRef<THREE.Mesh[]>([]);
  const goalMarkersRef = useRef<THREE.Mesh[]>([]);
  const sceneCenterRef = useRef<[number, number, number]>([0, 0, 0]);

  // Initialize scene
  useEffect(() => {
    if (!containerRef.current) return;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f5); // Light gray background
    sceneRef.current = scene;

    // Calculate scene center and size from bounds
    const bounds = metadata.bounds;
    const minX = bounds.min[0] ?? 0;
    const minY = bounds.min[1] ?? 0;
    const minZ = bounds.min[2] ?? 0;
    const maxX = bounds.max[0] ?? 10;
    const maxY = bounds.max[1] ?? 10;
    const maxZ = bounds.max[2] ?? 10;
    
    const sceneCenter: [number, number, number] = [
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    ];
    sceneCenterRef.current = sceneCenter;

    const size = Math.max(
      maxX - minX,
      maxY - minY,
      maxZ - minZ
    ) || 14; // Default to 14 if size is 0
    
    // Camera (orthographic for better spatial understanding)
    // Use scene size to determine viewSize, with some padding
    const viewSize = size * 1.2; // Make viewSize larger than scene to see everything
    const aspect = window.innerWidth / window.innerHeight;
    const camera = new THREE.OrthographicCamera(
      -viewSize * aspect,
      viewSize * aspect,
      viewSize,
      -viewSize,
      0.1,
      1000
    );
    cameraRef.current = camera;

    // Position camera closer with better angle to see objects
    const cameraDistance = size * 1.5;
    camera.position.set(
      sceneCenter[0] + cameraDistance * 0.6,
      sceneCenter[1] + cameraDistance * 0.6,
      sceneCenter[2] + cameraDistance
    );
    camera.lookAt(sceneCenter[0], sceneCenter[1], sceneCenter[2]);
    camera.updateProjectionMatrix();
   

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio); // Ensure proper pixel ratio
    renderer.shadowMap.enabled = true;
    const canvas = renderer.domElement;
    canvas.style.pointerEvents = "auto";
    canvas.style.cursor = "grab";
    canvas.style.display = "block";
    canvas.style.position = "absolute";
    canvas.style.top = "0";
    canvas.style.left = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.zIndex = "1";
    if (containerRef.current) {
      containerRef.current.appendChild(canvas);
      
    }
    rendererRef.current = renderer;

    // Simple orbit controls (basic implementation)
    let isMouseDown = false;
    let mouseX = 0;
    let mouseY = 0;

    const onMouseDown = (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      isMouseDown = true;
      mouseX = e.clientX;
      mouseY = e.clientY;
      renderer.domElement.style.cursor = "grabbing";
    };

    const onMouseUp = (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      isMouseDown = false;
      renderer.domElement.style.cursor = "grab";
    };

    const onMouseMove = (e: MouseEvent) => {
      if (isMouseDown && camera) {
        const deltaX = e.clientX - mouseX;
        const deltaY = e.clientY - mouseY;
        const sceneCenter = sceneCenterRef.current;
        const centerVec = new THREE.Vector3(
          sceneCenter[0],
          sceneCenter[1],
          sceneCenter[2]
        );
        const offset = new THREE.Vector3().subVectors(
          camera.position,
          centerVec
        );
        const spherical = new THREE.Spherical();
        spherical.setFromVector3(offset);
        spherical.theta -= deltaX * 0.01;
        spherical.phi += deltaY * 0.01;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
        offset.setFromSpherical(spherical);
        camera.position.copy(centerVec).add(offset);
        camera.lookAt(centerVec);
        camera.updateProjectionMatrix(); // Update projection matrix after rotation
        mouseX = e.clientX;
        mouseY = e.clientY;
      }
    };

    // Use document-level listeners for mouseup/mousemove so they work even if mouse leaves canvas
    const canvasElement = renderer.domElement;
    
    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      // Skip if clicking directly on a UI element
      const clickedUIElement = target.closest('#energy-bar') || 
                               target.closest('#goal-counter') || 
                               target.closest('#controls') || 
                               target.closest('#plots-toggle') ||
                               target.closest('#training-plots') ||
                               target.closest('button');
      if (clickedUIElement) {
        return; // Let UI elements handle their own clicks
      }
      // Handle clicks on canvas element - check if target is canvas or its parent
      if (target === canvasElement || target.tagName === 'CANVAS' || canvasElement.contains(target)) {
        e.stopPropagation();
        e.preventDefault();
        onMouseDown(e);
      }
    };
    
    const handleMouseUp = (e: MouseEvent) => {
      if (isMouseDown) {
        e.preventDefault();
        e.stopPropagation();
        onMouseUp(e);
      }
    };
    
    const handleMouseMove = (e: MouseEvent) => {
      if (isMouseDown) {
        e.preventDefault();
        e.stopPropagation();
        onMouseMove(e);
      }
    };
    
    // Attach to canvas element directly - use capture phase to get events first
    canvasElement.addEventListener("mousedown", handleMouseDown, { capture: true, passive: false });
    document.addEventListener("mouseup", handleMouseUp, { passive: false });
    document.addEventListener("mousemove", handleMouseMove, { passive: false });
    canvasElement.addEventListener("mouseleave", () => {
      isMouseDown = false;
      canvasElement.style.cursor = "grab";
    });

    // Mouse wheel zoom
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      
      if (!camera) return;
      
      const delta = e.deltaY * 0.001;
      const currentViewSize = (camera.right - camera.left) / (2 * aspect);
      const newViewSize = Math.max(1, Math.min(50, currentViewSize * (1 + delta)));
      
      camera.left = -newViewSize * aspect;
      camera.right = newViewSize * aspect;
      camera.top = newViewSize;
      camera.bottom = -newViewSize;
      camera.updateProjectionMatrix();
    };
    
    canvasElement.addEventListener("wheel", handleWheel, { passive: false });

    controlsRef.current = { update: () => {} };

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Handle window resize
    const onResize = () => {
      if (!camera || !renderer || !metadata) return;
      const size = Math.max(
        (metadata.bounds.max[0] ?? 10) - (metadata.bounds.min[0] ?? 0),
        (metadata.bounds.max[1] ?? 10) - (metadata.bounds.min[1] ?? 0),
        (metadata.bounds.max[2] ?? 10) - (metadata.bounds.min[2] ?? 0)
      );
      const viewSize = size * 1.2;
      const aspect = window.innerWidth / window.innerHeight;
      camera.left = -viewSize * aspect;
      camera.right = viewSize * aspect;
      camera.top = viewSize;
      camera.bottom = -viewSize;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener("resize", onResize);

    // Initialize static geometry
    initStaticGeometry(scene, metadata);
    // Animation loop
    let frameCount = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      if (controlsRef.current) {
        controlsRef.current.update();
      }
      if (renderer && scene && camera) {
        renderer.render(scene, camera);
        frameCount++;
      }
    };
    animate();

      return () => {
        window.removeEventListener("resize", onResize);
        canvasElement.removeEventListener("mousedown", handleMouseDown, { capture: true } as any);
        canvasElement.removeEventListener("wheel", handleWheel);
        document.removeEventListener("mouseup", handleMouseUp, { passive: false } as any);
        document.removeEventListener("mousemove", handleMouseMove, { passive: false } as any);
      if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
      // Dispose geometries and materials
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          if (Array.isArray(object.material)) {
            object.material.forEach((mat) => mat.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
      
      // IMPORTANT: Reset marker refs so they get recreated with the new scene
      currentPosMarkerRef.current = null;
      activeGoalMarkersRef.current = [];
      goalMarkersRef.current = [];
      dynamicObjectsRef.current = [];
      obstaclesRef.current = null;
      boundaryLinesRef.current = null;
      sceneRef.current = null;
    };
  }, [metadata]);

  // Initialize static geometry (obstacles, boundaries, start marker)
  const initStaticGeometry = (scene: THREE.Scene, metadata: Metadata) => {
    // Obstacles using InstancedMesh
    const obstacleGeometry = new THREE.SphereGeometry(1, 16, 16);
    const obstacleMaterial = new THREE.MeshStandardMaterial({
      color: COLORS.obstacle,
      transparent: true,
      opacity: 0.25,
    });

    const obstacleMesh = new THREE.InstancedMesh(
      obstacleGeometry,
      obstacleMaterial,
      metadata.obstacles.length
    );

    const matrix = new THREE.Matrix4();
    metadata.obstacles.forEach((obs, i) => {
      const [x, y, z, radius] = obs;
      matrix.makeScale(radius, radius, radius);
      matrix.setPosition(x, y, z);
      obstacleMesh.setMatrixAt(i, matrix);
    });

    obstacleMesh.instanceMatrix.needsUpdate = true;
    scene.add(obstacleMesh);
    obstaclesRef.current = obstacleMesh;

    // Boundaries (wireframe box)
    const bounds = metadata.bounds;
    const minX = bounds.min[0] ?? 0;
    const minY = bounds.min[1] ?? 0;
    const minZ = bounds.min[2] ?? 0;
    const maxX = bounds.max[0] ?? 10;
    const maxY = bounds.max[1] ?? 10;
    const maxZ = bounds.max[2] ?? 10;

    const boundaryGeometry = new THREE.BoxGeometry(
      maxX - minX,
      maxY - minY,
      maxZ - minZ
    );
    const boundaryEdges = new THREE.EdgesGeometry(boundaryGeometry);
    const boundaryMaterial = new THREE.LineBasicMaterial({
      color: COLORS.boundary,
      transparent: true,
      opacity: 0.4,
      linewidth: 1,
    });
    const boundaryLines = new THREE.LineSegments(
      boundaryEdges,
      boundaryMaterial
    );
    boundaryLines.position.set(
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2
    );
    scene.add(boundaryLines);
    boundaryLinesRef.current = boundaryLines;
  };

  // Clear dynamic geometry
  const clearDynamicGeometry = (scene: THREE.Scene) => {
    dynamicObjectsRef.current.forEach((obj) => scene.remove(obj));
    dynamicObjectsRef.current = [];
  };

  // Render tube (planned trajectory)
  const renderTube = (
    scene: THREE.Scene,
    mu_t: number[][],
    sigma_t: number[][],
    current_pos: number[],
    isCurrent: boolean
  ) => {
    if (!mu_t || !Array.isArray(mu_t) || mu_t.length < 2) {
      return;
    }

    // Convert relative to global coordinates
    const globalPoints = mu_t.map((pt) => [
      (pt[0] ?? 0) + (current_pos[0] ?? 0),
      (pt[1] ?? 0) + (current_pos[1] ?? 0),
      (pt[2] ?? 0) + (current_pos[2] ?? 0),
    ]);

    const points = globalPoints.map(
      (p) => new THREE.Vector3(p[0], p[1], p[2])
    );
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const opacity = isCurrent ? 0.85 : 0.5;
    const material = new THREE.LineBasicMaterial({
      color: COLORS.trajectory,
      linewidth: 1, // Note: linewidth doesn't work in WebGL, but we keep it for compatibility
      transparent: true,
      opacity: opacity,
    });
    const line = new THREE.Line(geometry, material);
    line.computeLineDistances();
    scene.add(line);
    dynamicObjectsRef.current.push(line);

    // Render capsule around tube
    if (sigma_t && Array.isArray(sigma_t) && sigma_t.length > 0) {
      const capsuleOpacity = isCurrent ? 0.25 : 0.15;
      renderCapsule(scene, globalPoints, sigma_t, COLORS.trajectory, capsuleOpacity);
    }
  };

  // Render actual path
  const renderPath = (
    scene: THREE.Scene,
    actual_path: number[][],
    sigma_t: number[][],
    isCurrent: boolean
  ) => {
    if (!actual_path || !Array.isArray(actual_path) || actual_path.length < 2) {
      return;
    }

    const points = actual_path.map(
      (p) => new THREE.Vector3(p[0], p[1], p[2] || 0)
    );
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const opacity = isCurrent ? 0.9 : 0.6;
    const material = new THREE.LineBasicMaterial({
      color: COLORS.trajectory,
      linewidth: 1, // Note: linewidth doesn't work in WebGL
      transparent: true,
      opacity: opacity,
    });
    const line = new THREE.Line(geometry, material);
    line.computeLineDistances();
    scene.add(line);
    dynamicObjectsRef.current.push(line);

    // Render capsule around path
    if (sigma_t && Array.isArray(sigma_t) && sigma_t.length > 0) {
      const capsuleOpacity = isCurrent ? 0.25 : 0.15;
      renderCapsule(scene, actual_path, sigma_t, COLORS.trajectory, capsuleOpacity);
    }
  };

  // Render anisotropic capsule (simplified approximation)
  const renderCapsule = (
    scene: THREE.Scene,
    pathPoints: number[][],
    sigmaPerDim: number[][],
    color: number,
    alpha: number
  ) => {
    if (!pathPoints || pathPoints.length < 2) {
      return;
    }
    if (!sigmaPerDim || !Array.isArray(sigmaPerDim) || sigmaPerDim.length === 0) {
      return;
    }

    // Calculate average radius per point
    const avgRadii = pathPoints.map((pt, i) => {
      if (i < sigmaPerDim.length) {
        if (Array.isArray(sigmaPerDim[i])) {
          // sigmaPerDim[i] is an array of per-dimension radii [x_radius, y_radius, z_radius]
          // Use the maximum radius for visibility
          return Math.max(...sigmaPerDim[i]);
        } else if (typeof sigmaPerDim[i] === "number") {
          // sigmaPerDim[i] is a single number
          return sigmaPerDim[i];
        }
      }
      return 0.1; // Default radius
    });

    const curve = new THREE.CatmullRomCurve3(
      pathPoints.map((p) => new THREE.Vector3(p[0], p[1], p[2] || 0))
    );

    // Use average of all radii, or default to 0.1
    const radius =
      avgRadii.length > 0
        ? avgRadii.reduce((a, b) => a + b, 0) / avgRadii.length
        : 0.1;
    
    // Ensure minimum radius for visibility - make it larger
    const finalRadius = Math.max(radius, 0.15);


    try {
      const tubeGeometry = new THREE.TubeGeometry(
        curve,
        Math.max(pathPoints.length * 2, 20),
        finalRadius,
        8,
        false
      );
      const tubeMaterial = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: alpha,
        side: THREE.DoubleSide,
      });
      const tube = new THREE.Mesh(tubeGeometry, tubeMaterial);
      scene.add(tube);
      dynamicObjectsRef.current.push(tube);
    } catch (error) {
      console.error("Error creating capsule:", error);
    }
  };

  // Update frame visualization
  useEffect(() => {
    if (
      !sceneRef.current ||
      !cameraRef.current ||
      currentFrameIndex < 0 ||
      currentFrameIndex >= frames.length
    )
      return;

    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const frame = frames[currentFrameIndex];


    // Clear previous dynamic geometry
    clearDynamicGeometry(scene);

    // Render history: show last N moves (tubes and paths)
    const historyStart = Math.max(0, currentFrameIndex - maxHistory + 1);
    
    for (let i = historyStart; i <= currentFrameIndex; i++) {
      const histFrame = frames[i];
      if (!histFrame) continue;
      
      const isCurrent = i === currentFrameIndex;
      
      // Render planned trajectory (mu_t)
      if (histFrame.mu_t && Array.isArray(histFrame.mu_t) && histFrame.mu_t.length > 0) {
        renderTube(
          scene,
          histFrame.mu_t,
          histFrame.sigma_t,
          histFrame.current_pos,
          isCurrent
        );
      }
      
      // Render actual path
      if (histFrame.actual_path && Array.isArray(histFrame.actual_path) && histFrame.actual_path.length > 0) {
        renderPath(
          scene,
          histFrame.actual_path,
          histFrame.sigma_t,
          isCurrent
        );
      }
    }
    

    // Update current position marker - blue orb (always visible)
    const pos = frame?.current_pos;
    
    if (pos && Array.isArray(pos) && pos.length >= 3) {
      const x = pos[0] ?? 0;
      const y = pos[1] ?? 0;
      const z = pos.length > 2 ? (pos[2] ?? 0) : 0;
      
      if (!currentPosMarkerRef.current) {
        // Create a prominent blue orb - 1.7x size of reached goals (0.18 * 1.7 = 0.306)
        const geometry = new THREE.SphereGeometry(0.3, 32, 32);
        const material = new THREE.MeshBasicMaterial({
          color: COLORS.current_pos, // Blue 0x3B82F6
          transparent: false,
        });
        const marker = new THREE.Mesh(geometry, material);
        marker.renderOrder = 1000; // Render on top
        marker.position.set(x, y, z);
        scene.add(marker);
        currentPosMarkerRef.current = marker;
        
        // Add a PointLight at the position for extra visibility
        const pointLight = new THREE.PointLight(COLORS.current_pos, 3, 20);
        pointLight.position.set(x, y, z);
        scene.add(pointLight);
        (marker as any).pointLight = pointLight;
        
        
      } else {
        currentPosMarkerRef.current.position.set(x, y, z);
        if ((currentPosMarkerRef.current as any).pointLight) {
          (currentPosMarkerRef.current as any).pointLight.position.set(x, y, z);
        }
      }
      currentPosMarkerRef.current.visible = true;
      if ((currentPosMarkerRef.current as any).pointLight) {
        (currentPosMarkerRef.current as any).pointLight.visible = true;
      }
    } else if (currentPosMarkerRef.current) {
      currentPosMarkerRef.current.visible = false;
      if ((currentPosMarkerRef.current as any).pointLight) {
        (currentPosMarkerRef.current as any).pointLight.visible = false;
      }
    }

    // Update active goal markers (render all active goals)
    const activeGoals = frame?.active_goals || [];
    
    // Remove old active goal markers
    activeGoalMarkersRef.current.forEach((marker) => {
      scene.remove(marker);
      if (marker.geometry) marker.geometry.dispose();
      if (marker.material) {
        if (Array.isArray(marker.material)) {
          marker.material.forEach((mat) => mat.dispose());
        } else {
          marker.material.dispose();
        }
      }
      // Remove point light if it exists
      if ((marker as any).pointLight) {
        scene.remove((marker as any).pointLight);
      }
    });
    activeGoalMarkersRef.current = [];
    
    // Create markers for all active goals
    activeGoals.forEach((goalPos) => {
      if (goalPos && Array.isArray(goalPos) && goalPos.length >= 3) {
        const x = goalPos[0] ?? 0;
        const y = goalPos[1] ?? 0;
        const z = goalPos.length > 2 ? (goalPos[2] ?? 0) : 0;
        
        // Create a prominent orange/yellow goal orb - 1.7x size of reached goals (0.18 * 1.7 = 0.306)
        const goalGeometry = new THREE.SphereGeometry(0.3, 32, 32);
        const goalMaterial = new THREE.MeshBasicMaterial({
          color: COLORS.goal, // Orange 0xF59E0B
          transparent: false,
        });
        const goalMarker = new THREE.Mesh(goalGeometry, goalMaterial);
        goalMarker.renderOrder = 1000; // Render on top
        goalMarker.position.set(x, y, z);
        scene.add(goalMarker);
        activeGoalMarkersRef.current.push(goalMarker);
        
        // Add a PointLight at the position for extra visibility
        const pointLight = new THREE.PointLight(COLORS.goal, 3, 20);
        pointLight.position.set(x, y, z);
        scene.add(pointLight);
        (goalMarker as any).pointLight = pointLight;
      }
    });

    // Render reached goals (only those reached up to this frame)
    goalMarkersRef.current.forEach((marker) => {
      scene.remove(marker);
      if (marker.geometry) marker.geometry.dispose();
      if (marker.material) {
        if (Array.isArray(marker.material)) {
          marker.material.forEach((mat) => mat.dispose());
        } else {
          marker.material.dispose();
        }
      }
    });
    goalMarkersRef.current = [];

    const reachedGoals = reachedGoalsByFrame[currentFrameIndex] || [];
    if (reachedGoals.length > 0) {
      reachedGoals.forEach((goalPos, idx) => {
        if (goalPos && Array.isArray(goalPos) && goalPos.length >= 3) {
          const x = goalPos[0] ?? 0;
          const y = goalPos[1] ?? 0;
          const z = goalPos.length > 2 ? (goalPos[2] ?? 0) : 0;
          // Wireframe with subtle background color
          const goalGeometry = new THREE.SphereGeometry(0.18, 16, 16);
          // Background sphere with low opacity
          const bgMaterial = new THREE.MeshBasicMaterial({
            color: COLORS.goal_reached, // Yellow/gold
            transparent: true,
            opacity: 0.3,
          });
          const bgSphere = new THREE.Mesh(goalGeometry, bgMaterial);
          bgSphere.position.set(x, y, z);
          scene.add(bgSphere);
          goalMarkersRef.current.push(bgSphere);
          
          // Wireframe overlay
          const wireframeGeometry = new THREE.SphereGeometry(0.18, 16, 16);
          const wireframeMaterial = new THREE.MeshBasicMaterial({
            color: COLORS.goal_reached,
            transparent: true,
            opacity: 0.8,
            wireframe: true,
          });
          const wireframeSphere = new THREE.Mesh(wireframeGeometry, wireframeMaterial);
          wireframeSphere.position.set(x, y, z);
          scene.add(wireframeSphere);
          goalMarkersRef.current.push(wireframeSphere);
        }
      });
    }
  }, [frames, currentFrameIndex, maxHistory, reachedGoalsByFrame]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full absolute top-0 left-0 z-[1]"
      style={{ pointerEvents: "auto" }}
    />
  );
}
