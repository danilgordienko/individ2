import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class Vec3:
    """3D вектор"""
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        length = self.length()
        if length > 0:
            return self / length
        return Vec3(0, 0, 0)

    def to_array(self):
        return np.array([self.x, self.y, self.z])


@dataclass
class Ray:
    """Луч"""
    origin: Vec3
    direction: Vec3


@dataclass
class Material:
    """Материал поверхности"""
    color: Vec3
    ambient: float = 0.1
    diffuse: float = 0.7
    specular: float = 0.2
    shininess: float = 32.0
    emissive: Vec3 = None
    transparency: float = 0.0

    def __post_init__(self):
        if self.emissive is None:
            self.emissive = Vec3(0, 0, 0)


@dataclass
class HitRecord:
    """Информация о пересечении луча с объектом"""
    point: Vec3
    normal: Vec3
    t: float
    material: Material


class Sphere:
    """Сфера"""

    def __init__(self, center: Vec3, radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material
        self.original_material = material

    def set_transparency(self, transparent: bool):
        """Включение/выключение прозрачности"""
        if transparent:
            self.material = Material(
                color=self.original_material.color,
                ambient=self.original_material.ambient,
                diffuse=self.original_material.diffuse,
                specular=self.original_material.specular,
                shininess=self.original_material.shininess,
                emissive=self.original_material.emissive,
                transparency=0.6
            )
        else:
            self.material = self.original_material

    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t = (-b - math.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:
            t = (-b + math.sqrt(discriminant)) / (2.0 * a)
            if t < 0.001:
                return None

        point = Vec3(
            ray.origin.x + t * ray.direction.x,
            ray.origin.y + t * ray.direction.y,
            ray.origin.z + t * ray.direction.z
        )
        normal = (point - self.center).normalize()

        return HitRecord(point, normal, t, self.material)


class Cube:
    """Куб"""

    def __init__(self, center: Vec3, size: float, material: Material):
        self.center = center
        self.size = size
        self.material = material
        self.original_material = material
        self.min_corner = Vec3(center.x - size / 2, center.y - size / 2, center.z - size / 2)
        self.max_corner = Vec3(center.x + size / 2, center.y + size / 2, center.z + size / 2)

    def set_transparency(self, transparent: bool):
        """Включение/выключение прозрачности"""
        if transparent:
            self.material = Material(
                color=self.original_material.color,
                ambient=self.original_material.ambient,
                diffuse=self.original_material.diffuse,
                specular=self.original_material.specular,
                shininess=self.original_material.shininess,
                emissive=self.original_material.emissive,
                transparency=0.6
            )
        else:
            self.material = self.original_material

    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        tmin = (self.min_corner.x - ray.origin.x) / ray.direction.x if ray.direction.x != 0 else float('-inf')
        tmax = (self.max_corner.x - ray.origin.x) / ray.direction.x if ray.direction.x != 0 else float('inf')

        if tmin > tmax:
            tmin, tmax = tmax, tmin

        tymin = (self.min_corner.y - ray.origin.y) / ray.direction.y if ray.direction.y != 0 else float('-inf')
        tymax = (self.max_corner.y - ray.origin.y) / ray.direction.y if ray.direction.y != 0 else float('inf')

        if tymin > tymax:
            tymin, tymax = tymax, tymin

        if (tmin > tymax) or (tymin > tmax):
            return None

        if tymin > tmin:
            tmin = tymin
        if tymax < tmax:
            tmax = tymax

        tzmin = (self.min_corner.z - ray.origin.z) / ray.direction.z if ray.direction.z != 0 else float('-inf')
        tzmax = (self.max_corner.z - ray.origin.z) / ray.direction.z if ray.direction.z != 0 else float('inf')

        if tzmin > tzmax:
            tzmin, tzmax = tzmax, tzmin

        if (tmin > tzmax) or (tzmin > tmax):
            return None

        if tzmin > tmin:
            tmin = tzmin
        if tzmax < tmax:
            tmax = tzmax

        if tmin < 0.001:
            if tmax < 0.001:
                return None
            t = tmax
        else:
            t = tmin

        if t < 0.001:
            return None

        point = Vec3(
            ray.origin.x + t * ray.direction.x,
            ray.origin.y + t * ray.direction.y,
            ray.origin.z + t * ray.direction.z
        )

        normal = self.get_normal(point)

        return HitRecord(point, normal, t, self.material)

    def get_normal(self, point: Vec3) -> Vec3:
        epsilon = 0.001
        if abs(point.x - self.min_corner.x) < epsilon:
            return Vec3(-1, 0, 0)
        elif abs(point.x - self.max_corner.x) < epsilon:
            return Vec3(1, 0, 0)
        elif abs(point.y - self.min_corner.y) < epsilon:
            return Vec3(0, -1, 0)
        elif abs(point.y - self.max_corner.y) < epsilon:
            return Vec3(0, 1, 0)
        elif abs(point.z - self.min_corner.z) < epsilon:
            return Vec3(0, 0, -1)
        else:
            return Vec3(0, 0, 1)


class Plane:
    """Плоскость (для стен комнаты)"""

    def __init__(self, point: Vec3, normal: Vec3, material: Material):
        self.point = point
        self.normal = normal.normalize()
        self.material = material

    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        denom = self.normal.dot(ray.direction)
        if abs(denom) > 0.0001:
            t = (self.point - ray.origin).dot(self.normal) / denom
            if t > 0.001:
                point = Vec3(
                    ray.origin.x + t * ray.direction.x,
                    ray.origin.y + t * ray.direction.y,
                    ray.origin.z + t * ray.direction.z
                )
                return HitRecord(point, self.normal, t, self.material)
        return None


class PointLight:
    """Точечный источник света"""

    def __init__(self, position: Vec3, color: Vec3, intensity: float = 1.0):
        self.position = position
        self.color = color
        self.intensity = intensity


class Scene:
    """Сцена"""

    def __init__(self):
        self.objects = []
        self.lights = []
        self.transparent_mode = False

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light: PointLight):
        self.lights.append(light)

    def set_transparency(self, transparent: bool):
        """Установить прозрачность для всех объектов, которые могут быть прозрачными"""
        self.transparent_mode = transparent
        for obj in self.objects:
            if hasattr(obj, 'set_transparency'):
                obj.set_transparency(transparent)

    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        """Пересечение луча со сценой"""
        closest_hit = None
        closest_t = float('inf')

        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and hit.t < closest_t:
                closest_hit = hit
                closest_t = hit.t

        return closest_hit


def create_cornell_box() -> Scene:
    """Создание Корнуэльской комнаты"""
    scene = Scene()

    # Размер комнаты
    room_size = 5.0

    # Пол (белый)
    floor_material = Material(
        Vec3(0.73, 0.73, 0.73),
        ambient=0.3,
        diffuse=0.9,
        specular=0.1
    )
    scene.add_object(Plane(Vec3(0, -room_size / 2, 0), Vec3(0, 1, 0), floor_material))

    # Потолок (белый)
    ceiling_material = Material(
        Vec3(0.73, 0.73, 0.73),
        ambient=0.3,
        diffuse=0.9,
        specular=0.1
    )
    scene.add_object(Plane(Vec3(0, room_size / 2, 0), Vec3(0, -1, 0), ceiling_material))

    # Задняя стена (белая)
    back_wall_material = Material(
        Vec3(0.73, 0.73, 0.73),
        ambient=0.3,
        diffuse=0.9,
        specular=0.1
    )
    scene.add_object(Plane(Vec3(0, 0, -room_size / 2), Vec3(0, 0, 1), back_wall_material))

    # Левая стена (красная)
    left_wall_material = Material(
        Vec3(0.65, 0.05, 0.05),
        ambient=0.3,
        diffuse=0.9,
        specular=0.1
    )
    scene.add_object(Plane(Vec3(-room_size / 2, 0, 0), Vec3(1, 0, 0), left_wall_material))

    # Правая стена (синяя)
    blue_wall_material = Material(
        Vec3(0.05, 0.05, 0.65),
        ambient=0.3,
        diffuse=0.9,
        specular=0.1
    )
    scene.add_object(Plane(Vec3(room_size / 2, 0, 0), Vec3(-1, 0, 0), blue_wall_material))

    # Желтая сфера
    yellow_material = Material(
        Vec3(0.8, 0.8, 0.1),
        ambient=0.2,
        diffuse=0.8,
        specular=0.3,
        shininess=64.0
    )
    yellow_sphere = Sphere(Vec3(-1.2, -1.5, -0.5), 0.7, yellow_material)
    scene.add_object(yellow_sphere)

    # Бежевая сфера
    beige_material = Material(
        Vec3(0.6, 0.5, 0.4),
        ambient=0.2,
        diffuse=0.8,
        specular=0.2,
        shininess=32.0
    )
    beige_sphere = Sphere(Vec3(1.0, -0.5, 0.5), 1.0, beige_material)
    scene.add_object(beige_sphere)

    # Куб
    purple_material = Material(
        Vec3(0.5, 0.1, 0.7),
        ambient=0.2,
        diffuse=0.7,
        specular=0.4,
        shininess=128.0
    )
    cube = Cube(Vec3(0, -2.0, 3.0), 1.0, purple_material)
    scene.add_object(cube)

    # источник света
    light_material = Material(
        Vec3(1, 1, 1),
        ambient=0.0,
        diffuse=0.0,
        specular=0.0,
        emissive=Vec3(8.0, 8.0, 8.0)
    )
    scene.add_object(Sphere(Vec3(0, 2.3, 0), 0.4, light_material))

    # Точечный источник света
    scene.add_light(PointLight(Vec3(0, 2.3, 0), Vec3(1, 1, 1), 30.0))

    # Дополнительный свет
    scene.add_light(PointLight(Vec3(0, 1.5, 0), Vec3(1, 1, 1), 15.0))

    return scene


def shade_simple_transparency(hit: HitRecord, scene: Scene, ray: Ray) -> Vec3:
    """Простая версия затенения с прозрачностью"""
    # Если объект светится сам
    if hit.material.emissive.x > 0 or hit.material.emissive.y > 0 or hit.material.emissive.z > 0:
        return hit.material.emissive

    color = Vec3(0, 0, 0)

    # Ambient (окружающее освещение)
    ambient = hit.material.color * hit.material.ambient
    color = color + ambient

    for light in scene.lights:
        # Направление к источнику света
        light_dir = (light.position - hit.point).normalize()
        light_distance = (light.position - hit.point).length()

        # Аттенюация по расстоянию
        attenuation = 1.0 / (light_distance * light_distance + 1.0)

        # Проверка на тень
        shadow_ray = Ray(hit.point + hit.normal * 0.001, light_dir)
        shadow_hit = scene.intersect(shadow_ray)

        # Для прозрачных объектов делаем тени слабее
        shadow_factor = 1.0
        if shadow_hit and shadow_hit.t < light_distance - 0.001:
            if shadow_hit.material.transparency > 0:
                shadow_factor = 1.0 - shadow_hit.material.transparency * 0.3
            else:
                continue

        # Diffuse (диффузное отражение)
        diffuse_intensity = max(0, hit.normal.dot(light_dir))
        if diffuse_intensity > 0:
            diffuse_color = Vec3(
                hit.material.color.x * light.color.x,
                hit.material.color.y * light.color.y,
                hit.material.color.z * light.color.z
            )
            diffuse = diffuse_color * hit.material.diffuse * diffuse_intensity * light.intensity * attenuation * shadow_factor
            color = color + diffuse

        # Specular (зеркальное отражение)
        view_dir = (ray.origin - hit.point).normalize()
        half_dir = (light_dir + view_dir).normalize()
        spec_intensity = max(0, hit.normal.dot(half_dir)) ** hit.material.shininess
        if spec_intensity > 0:
            specular = light.color * hit.material.specular * spec_intensity * light.intensity * attenuation * shadow_factor
            color = color + specular

    # Для прозрачных объектов добавляем цвет фона, чтобы было видно сквозь них
    if hit.material.transparency > 0:
        # Продолжаем луч чтобы найти фон
        background_ray = Ray(hit.point + ray.direction * 0.001, ray.direction)
        background_hit = scene.intersect(background_ray)

        if background_hit:
            background_color = shade_simple_transparency(background_hit, scene, background_ray)
            # Смешиваем цвет объекта с цветом фона
            transparency = hit.material.transparency
            color = Vec3(
                color.x * (1 - transparency) + background_color.x * transparency,
                color.y * (1 - transparency) + background_color.y * transparency,
                color.z * (1 - transparency) + background_color.z * transparency
            )
        else:
            # Если нет фона, просто делаем объект полупрозрачным
            transparency = hit.material.transparency
            color = color * (1 - transparency * 0.5)

    # Гамма-коррекция
    gamma = 2.2
    color = Vec3(
        min(1.0, color.x ** (1.0 / gamma)),
        min(1.0, color.y ** (1.0 / gamma)),
        min(1.0, color.z ** (1.0 / gamma))
    )

    return color


def render(scene: Scene, width: int = 800, height: int = 600, anti_aliasing: bool = True) -> np.ndarray:
    """Рендеринг сцены с антиалиасингом"""
    image = np.zeros((height, width, 3))

    camera_pos = Vec3(0, 0, 10.0)
    aspect_ratio = width / height
    fov = 45
    scale = math.tan(math.radians(fov * 0.5))

    samples_per_pixel = 4 if anti_aliasing else 1

    for y in range(height):
        if y % 50 == 0:
            print(f"Рендеринг: {y}/{height}")

        for x in range(width):
            pixel_color = Vec3(0, 0, 0)

            for sample in range(samples_per_pixel):
                # Добавляем случайное смещение для антиалиасинга
                if anti_aliasing and samples_per_pixel > 1:
                    offset_x = (np.random.random() - 0.5) * 0.5
                    offset_y = (np.random.random() - 0.5) * 0.5
                else:
                    offset_x = offset_y = 0

                # Вычисление направления луча
                px = (2 * (x + 0.5 + offset_x) / width - 1) * aspect_ratio * scale
                py = (1 - 2 * (y + 0.5 + offset_y) / height) * scale

                ray_dir = Vec3(px, py, -1).normalize()
                ray = Ray(camera_pos, ray_dir)

                # Трассировка луча
                hit = scene.intersect(ray)

                if hit:
                    color = shade_simple_transparency(hit, scene, ray)
                    pixel_color = pixel_color + color

            # Усредняем по сэмплам
            if samples_per_pixel > 1:
                pixel_color = pixel_color / samples_per_pixel

            image[y, x] = [pixel_color.x, pixel_color.y, pixel_color.z]

    return image


class InteractiveRenderer:
    """Интерактивный рендерер с кнопкой"""

    def __init__(self):
        self.scene = create_cornell_box()
        self.transparent_mode = False
        self.render_width = 800
        self.render_height = 600

        # Создаем фигуру с кнопкой
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        plt.subplots_adjust(bottom=0.15)

        # Создаем кнопку
        self.button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.button = Button(self.button_ax, 'Включить прозрачность',
                             color='lightgoldenrodyellow', hovercolor='0.975')

        # Назначаем обработчик нажатия
        self.button.on_clicked(self.toggle_transparency)

        # Первоначальный рендеринг
        self.render_and_display()

    def toggle_transparency(self, event):
        """Переключение режима прозрачности"""
        self.transparent_mode = not self.transparent_mode

        # Обновляем сцену
        self.scene.set_transparency(self.transparent_mode)

        # Обновляем текст кнопки
        if self.transparent_mode:
            self.button.label.set_text('Выключить прозрачность')
        else:
            self.button.label.set_text('Включить прозрачность')

        # Перерисовываем изображение
        self.render_and_display()

    def render_and_display(self):
        """Рендеринг и отображение сцены"""
        print(f"Рендеринг (прозрачность: {'ВКЛ' if self.transparent_mode else 'ВЫКЛ'})...")

        # Рендерим сцену с высоким качеством и антиалиасингом
        image = render(self.scene,
                       width=self.render_width,
                       height=self.render_height,
                       anti_aliasing=True)

        # Очищаем и отображаем новое изображение
        self.ax.clear()
        self.ax.imshow(image, interpolation='bicubic')  # Качественная интерполяция
        self.ax.axis('off')

        title = 'Корнуэльская комната с кубом'
        if self.transparent_mode:
            title += ' (прозрачность ВКЛ)'
        self.ax.set_title(title, fontsize=16, pad=20)

        # Обновляем отображение
        self.fig.canvas.draw_idle()
        print("Готово!")


def main():
    """Главная функция"""
    # Создаем интерактивный рендерер
    renderer = InteractiveRenderer()

    plt.show()


if __name__ == "__main__":
    main()
