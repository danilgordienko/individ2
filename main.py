"""
Корнуэльская комната с ray tracing
Управление:
  [1] - включить/выключить отражение шаров
  [2] - включить/выключить отражение кубов
  [3] - включить/выключить прозрачность шаров
  [4] - включить/выключить прозрачность кубов
  [L/R/B/T/F/N] - выбор зеркальной стены
  [SPACE] - перерендерить
"""

import tkinter as tk
import math

# ==================== МАТЕМАТИКА ====================

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        l = self.length()
        if l > 0:
            return Vec3(self.x/l, self.y/l, self.z/l)
        return Vec3(0, 0, 0)

    def reflect(self, normal):
        d = 2 * self.dot(normal)
        return Vec3(self.x - d * normal.x,
                   self.y - d * normal.y,
                   self.z - d * normal.z)

# ==================== ОБЪЕКТЫ ====================

class Material:
    def __init__(self, color, reflective=False, transparent=False):
        self.color = color
        self.reflective = reflective
        self.transparent = transparent

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray_origin, ray_dir):
        oc = ray_origin - self.center
        a = ray_dir.dot(ray_dir)
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return None

        t = (-b - math.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:
            return None

        point = ray_origin + ray_dir * t
        normal = (point - self.center).normalize()
        return (t, point, normal)

class Box:
    def __init__(self, center, size, material):
        self.center = center
        self.size = size
        self.material = material
        self.min_p = Vec3(center.x - size.x/2, center.y - size.y/2, center.z - size.z/2)
        self.max_p = Vec3(center.x + size.x/2, center.y + size.y/2, center.z + size.z/2)

    def intersect(self, ray_origin, ray_dir):
        tmin = (self.min_p.x - ray_origin.x) / ray_dir.x if abs(ray_dir.x) > 0.0001 else -1e10
        tmax = (self.max_p.x - ray_origin.x) / ray_dir.x if abs(ray_dir.x) > 0.0001 else 1e10

        if tmin > tmax:
            tmin, tmax = tmax, tmin

        tymin = (self.min_p.y - ray_origin.y) / ray_dir.y if abs(ray_dir.y) > 0.0001 else -1e10
        tymax = (self.max_p.y - ray_origin.y) / ray_dir.y if abs(ray_dir.y) > 0.0001 else 1e10

        if tymin > tymax:
            tymin, tymax = tymax, tymin

        if tmin > tymax or tymin > tmax:
            return None

        tmin = max(tmin, tymin)
        tmax = min(tmax, tymax)

        tzmin = (self.min_p.z - ray_origin.z) / ray_dir.z if abs(ray_dir.z) > 0.0001 else -1e10
        tzmax = (self.max_p.z - ray_origin.z) / ray_dir.z if abs(ray_dir.z) > 0.0001 else 1e10

        if tzmin > tzmax:
            tzmin, tzmax = tzmax, tzmin

        if tmin > tzmax or tzmin > tmax:
            return None

        tmin = max(tmin, tzmin)

        if tmin < 0.001:
            return None

        point = ray_origin + ray_dir * tmin

        epsilon = 0.0001
        if abs(point.x - self.min_p.x) < epsilon:
            normal = Vec3(-1, 0, 0)
        elif abs(point.x - self.max_p.x) < epsilon:
            normal = Vec3(1, 0, 0)
        elif abs(point.y - self.min_p.y) < epsilon:
            normal = Vec3(0, -1, 0)
        elif abs(point.y - self.max_p.y) < epsilon:
            normal = Vec3(0, 1, 0)
        elif abs(point.z - self.min_p.z) < epsilon:
            normal = Vec3(0, 0, -1)
        else:
            normal = Vec3(0, 0, 1)

        return (tmin, point, normal)

class Plane:
    def __init__(self, point, normal, material, wall_id=""):
        self.point = point
        self.normal = normal.normalize()
        self.material = material
        self.wall_id = wall_id

    def intersect(self, ray_origin, ray_dir):
        denom = self.normal.dot(ray_dir)
        if abs(denom) < 0.0001:
            return None

        t = (self.point - ray_origin).dot(self.normal) / denom
        if t < 0.001:
            return None

        point = ray_origin + ray_dir * t
        return (t, point, self.normal)

# ==================== RAY TRACER ====================

class RayTracer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.camera_pos = Vec3(0, 0, -5)
        self.light_pos = Vec3(0, 1.8, 0)
        self.objects = []

    def setup_scene(self, sphere_refl, cube_refl, sphere_trans, cube_trans, mirror_wall):
        self.objects = []

        # Стены
        walls = [
            Plane(Vec3(0, -2, 0), Vec3(0, 1, 0), Material((0.9, 0.9, 0.9)), "bottom"),
            Plane(Vec3(0, 2, 0), Vec3(0, -1, 0), Material((0.9, 0.9, 0.9)), "top"),
            Plane(Vec3(-2, 0, 0), Vec3(1, 0, 0), Material((0.9, 0.1, 0.1)), "left"),
            Plane(Vec3(2, 0, 0), Vec3(-1, 0, 0), Material((0.1, 0.9, 0.1)), "right"),
            Plane(Vec3(0, 0, 2), Vec3(0, 0, -1), Material((0.9, 0.9, 0.9)), "back"),
        ]

        for wall in walls:
            if wall.wall_id == mirror_wall:
                wall.material.reflective = True

        self.objects.extend(walls)

        # Шары
        self.objects.append(Sphere(
            Vec3(-0.7, -1.3, 0.5), 0.7,
            Material((0.3, 0.3, 0.9), sphere_refl, sphere_trans)
        ))
        self.objects.append(Sphere(
            Vec3(0.9, -1.2, -0.3), 0.8,
            Material((0.9, 0.9, 0.1), sphere_refl, sphere_trans)
        ))

        # Кубы
        self.objects.append(Box(
            Vec3(-0.8, 0.2, -0.5), Vec3(0.8, 1.2, 0.8),
            Material((0.9, 0.9, 0.9), cube_refl, cube_trans)
        ))
        self.objects.append(Box(
            Vec3(0.7, 0.5, 0.7), Vec3(0.9, 1.5, 0.9),
            Material((0.9, 0.9, 0.9), cube_refl, cube_trans)
        ))

    def trace(self, ray_origin, ray_dir, depth=0):
        if depth > 2:
            return (0.05, 0.05, 0.05)

        closest = None
        min_t = 1e10
        closest_obj = None

        for obj in self.objects:
            hit = obj.intersect(ray_origin, ray_dir)
            if hit and hit[0] < min_t:
                min_t = hit[0]
                closest = hit
                closest_obj = obj

        if not closest:
            return (0.05, 0.05, 0.05)

        t, point, normal = closest
        mat = closest_obj.material

        to_light = (self.light_pos - point).normalize()
        diffuse = max(0, normal.dot(to_light))

        shadow_ray_origin = point + normal * 0.001
        in_shadow = False
        for obj in self.objects:
            if obj != closest_obj and obj.intersect(shadow_ray_origin, to_light):
                in_shadow = True
                break

        ambient = 0.2
        light_intensity = ambient if in_shadow else ambient + diffuse * 1.2
        color = (mat.color[0] * light_intensity,
                mat.color[1] * light_intensity,
                mat.color[2] * light_intensity)

        if mat.reflective and depth < 2:
            reflect_dir = ray_dir.reflect(normal)
            reflect_origin = point + normal * 0.001
            reflect_color = self.trace(reflect_origin, reflect_dir, depth + 1)
            color = (color[0] * 0.3 + reflect_color[0] * 0.7,
                    color[1] * 0.3 + reflect_color[1] * 0.7,
                    color[2] * 0.3 + reflect_color[2] * 0.7)

        if mat.transparent and depth < 2:
            refract_origin = point + ray_dir * 0.001
            refract_color = self.trace(refract_origin, ray_dir, depth + 1)
            color = (color[0] * 0.3 + refract_color[0] * 0.7,
                    color[1] * 0.3 + refract_color[1] * 0.7,
                    color[2] * 0.3 + refract_color[2] * 0.7)

        return color

# ==================== GUI ====================

class CornellBoxApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Корнуэльская комната")
        self.root.geometry("850x650")
        self.root.configure(bg="#1a1a2e")

        self.width = 400  # Уменьшено для скорости
        self.height = 400

        self.sphere_refl = False
        self.cube_refl = False
        self.sphere_trans = False
        self.cube_trans = False
        self.mirror_wall = None

        self.rendering = False
        self.current_line = 0

        self.create_widgets()
        self.root.after(100, self.start_render)

    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(main_frame, width=400, height=400, bg="#0f0f1e",
                               highlightthickness=2, highlightbackground="#444")
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))

        control_frame = tk.Frame(main_frame, bg="#16213e", width=220)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        control_frame.pack_propagate(False)

        tk.Label(control_frame, text="Корнуэльская комната",
                bg="#16213e", fg="#e8e8e8",
                font=("Arial", 12, "bold")).pack(pady=(20, 15))

        tk.Label(control_frame, text="Отражение:",
                bg="#16213e", fg="#a8a8a8",
                font=("Arial", 10, "bold")).pack(pady=(10, 5))

        self.sphere_refl_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Шары",
                      variable=self.sphere_refl_var,
                      command=self.on_change,
                      bg="#16213e", fg="#e8e8e8",
                      selectcolor="#0f3460").pack()

        self.cube_refl_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Кубы",
                      variable=self.cube_refl_var,
                      command=self.on_change,
                      bg="#16213e", fg="#e8e8e8",
                      selectcolor="#0f3460").pack()

        tk.Label(control_frame, text="Прозрачность:",
                bg="#16213e", fg="#a8a8a8",
                font=("Arial", 10, "bold")).pack(pady=(15, 5))

        self.sphere_trans_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Шары",
                      variable=self.sphere_trans_var,
                      command=self.on_change,
                      bg="#16213e", fg="#e8e8e8",
                      selectcolor="#0f3460").pack()

        self.cube_trans_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Кубы",
                      variable=self.cube_trans_var,
                      command=self.on_change,
                      bg="#16213e", fg="#e8e8e8",
                      selectcolor="#0f3460").pack()

        tk.Label(control_frame, text="Зеркальная стена:",
                bg="#16213e", fg="#a8a8a8",
                font=("Arial", 10, "bold")).pack(pady=(15, 5))

        self.mirror_var = tk.StringVar(value="none")
        for text, value in [("Нет", "none"), ("Левая", "left"),
                           ("Правая", "right"), ("Задняя", "back"),
                           ("Пол", "bottom"), ("Потолок", "top")]:
            tk.Radiobutton(control_frame, text=text, variable=self.mirror_var,
                          value=value, command=self.on_change,
                          bg="#16213e", fg="#e8e8e8",
                          selectcolor="#0f3460").pack()

        tk.Button(control_frame, text="Рендер (Space)",
                 command=self.start_render,
                 bg="#0f3460", fg="#e8e8e8",
                 font=("Arial", 10, "bold")).pack(pady=20)

        self.status = tk.Label(control_frame, text="Готов",
                              bg="#16213e", fg="#4ecca3",
                              font=("Arial", 9))
        self.status.pack()

        self.root.bind("<space>", lambda e: self.start_render())
        self.root.bind("1", lambda e: self.toggle_var(self.sphere_refl_var))
        self.root.bind("2", lambda e: self.toggle_var(self.cube_refl_var))
        self.root.bind("3", lambda e: self.toggle_var(self.sphere_trans_var))
        self.root.bind("4", lambda e: self.toggle_var(self.cube_trans_var))

    def toggle_var(self, var):
        var.set(not var.get())
        self.on_change()

    def on_change(self):
        self.start_render()

    def start_render(self):
        if self.rendering:
            return

        self.rendering = True
        self.current_line = 0
        self.status.config(text="Рендеринг...", fg="#ffd700")

        self.img = tk.PhotoImage(width=self.width, height=self.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img)

        self.tracer = RayTracer(self.width, self.height)
        self.tracer.setup_scene(
            self.sphere_refl_var.get(),
            self.cube_refl_var.get(),
            self.sphere_trans_var.get(),
            self.cube_trans_var.get(),
            self.mirror_var.get() if self.mirror_var.get() != "none" else None
        )

        self.render_lines()

    def render_lines(self):
        if self.current_line >= self.height:
            self.rendering = False
            self.status.config(text="Готово!", fg="#4ecca3")
            return

        # Рендер нескольких строк за раз
        batch_size = 10
        for _ in range(batch_size):
            if self.current_line >= self.height:
                break

            y = self.current_line
            aspect = self.width / self.height
            scale = math.tan(math.radians(30))

            colors = []
            for x in range(self.width):
                px = (2 * (x + 0.5) / self.width - 1) * aspect * scale
                py = (1 - 2 * (y + 0.5) / self.height) * scale

                ray_dir = Vec3(px, py, 1).normalize()
                color = self.tracer.trace(self.tracer.camera_pos, ray_dir)

                r = min(255, int(color[0] * 255))
                g = min(255, int(color[1] * 255))
                b = min(255, int(color[2] * 255))
                colors.append(f"#{r:02x}{g:02x}{b:02x}")

            self.img.put(" ".join(colors), to=(0, y))
            self.current_line += 1

        progress = int((self.current_line / self.height) * 100)
        self.status.config(text=f"Рендеринг: {progress}%")

        self.root.after(1, self.render_lines)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CornellBoxApp()
    app.run()