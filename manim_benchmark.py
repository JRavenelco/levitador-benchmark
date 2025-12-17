"""
Videos explicativos del Benchmark del Levitador Magnético.
Versión Final (Consolidada) - Generados con Manim.

Contenido:
1. Scene1_Contexto: Ingeniería Inversa del sistema real.
2. Scene2_Fisica: Dinámica del modelo y ecuación L(y).
3. Scene3_Optimizacion: Visualización del ajuste de curvas (MSE).

Ejecutar para generar videos:
    manim -pql manim_benchmark.py Scene1_Contexto
    manim -pql manim_benchmark.py Scene2_Fisica
    manim -pql manim_benchmark.py Scene3_Optimizacion
"""

from manim import *
import numpy as np

# --- CONFIGURACIÓN ---
config.background_color = "#1e1e1e"

class Theme:
    K0 = "#4AF626"   # Verde neón
    K  = "#FFDD33"   # Amarillo
    A  = "#FF3355"   # Rojo rosado
    Y  = "#33CCFF"   # Azul claro
    REAL = "#4AF626" # Verde (Ground Truth)
    SIM  = "#FF3355" # Rojo (Simulación)
    TEXT = "#EEEEEE" # Blanco humo

def _title(scene, text, subtext=""):
    group = VGroup()
    t = Text(text, font_size=40, weight=BOLD, color=Theme.TEXT).to_edge(UP, buff=0.2)
    group.add(t)
    scene.play(Write(t))
    if subtext:
        s = Text(subtext, font_size=24, color=GRAY).next_to(t, DOWN, buff=0.1)
        group.add(s)
        scene.play(FadeIn(s, shift=UP))
    return group

# =============================================================================
# ESCENA 1: CONTEXTO (Ingeniería Inversa del Sistema)
# =============================================================================

class Scene1_Contexto(Scene):
    def construct(self):
        title_group = _title(self, "El Problema: Ingeniería Inversa", "Tenemos los datos, nos falta el modelo")
        
        # 1. Mostrar el sistema real
        box_ctrl = Rectangle(width=2.5, height=1.5, color=BLUE, fill_opacity=0.2)
        label_ctrl = Text("Controlador\n(Ya existe)", font_size=20).move_to(box_ctrl)
        
        box_plant = Rectangle(width=2.5, height=1.5, color=ORANGE, fill_opacity=0.2).next_to(box_ctrl, RIGHT, buff=2)
        label_plant = Text("Levitador\n(Planta Real)", font_size=20).move_to(box_plant)
        
        arrow1 = Arrow(box_ctrl.get_right(), box_plant.get_left(), color=WHITE)
        label_u = MathTex("u(t)", color=YELLOW, font_size=24).next_to(arrow1, UP, buff=0.1)
        sub_u = Text("Voltaje", font_size=16, color=YELLOW).next_to(arrow1, DOWN, buff=0.1)
        
        arrow2 = Arrow(box_plant.get_right(), box_plant.get_right() + RIGHT*1.5, color=WHITE)
        label_y = MathTex("y(t)", color=GREEN, font_size=24).next_to(arrow2, UP, buff=0.1)
        sub_y = Text("Posición", font_size=16, color=GREEN).next_to(arrow2, DOWN, buff=0.1)
        
        diagram = VGroup(box_ctrl, label_ctrl, box_plant, label_plant, arrow1, label_u, sub_u, arrow2, label_y, sub_y)
        diagram.move_to(ORIGIN)
        
        self.play(Create(box_ctrl), Write(label_ctrl))
        self.play(Create(box_plant), Write(label_plant))
        self.play(GrowArrow(arrow1), Write(label_u), FadeIn(sub_u))
        self.play(GrowArrow(arrow2), Write(label_y), FadeIn(sub_y))
        self.wait(1)
        
        # 2. El Misterio
        mystery = Text("?", font_size=60, color=RED).move_to(box_plant)
        self.play(Transform(label_plant, mystery))
        
        narrativa = Text(
            "Conocemos la entrada u(t) y la salida y(t).\nPero NO sabemos los parámetros físicos internos.",
            font_size=24, color=WHITE
        ).to_edge(DOWN)
        self.play(Write(narrativa))
        self.wait(3)
        
        # 3. Objetivo
        self.play(FadeOut(diagram), FadeOut(narrativa))
        
        objetivo = Text("Objetivo del Benchmark:", font_size=32, color=Theme.Y)
        objetivo.shift(UP)
        
        meta = Text(
            "Encontrar los valores físicos (k₀, k, a)\nque hacen que el modelo matemático\nse comporte IGUAL al sistema real.",
            font_size=28, line_spacing=1.2
        ).next_to(objetivo, DOWN)
        
        self.play(Write(objetivo))
        self.play(Write(meta))
        self.wait(3)


# =============================================================================
# ESCENA 2: FÍSICA (Abriendo la Caja Negra - Dinámico)
# =============================================================================

class Scene2_Fisica(Scene):
    def construct(self):
        # Título arriba
        t_group = _title(self, "El Modelo Físico", "¿Qué ecuaciones gobiernan el sistema?")
        
        # --- IZQUIERDA: Sistema Mecánico ---
        mech_group = VGroup()
        # Posicionamos el imán más abajo y a la izquierda
        magnet = Rectangle(width=2, height=1, color=GRAY, fill_opacity=0.5).move_to(LEFT*3.5 + UP*0.5)
        coil = VGroup(*[
            Ellipse(width=2.2, height=0.5, color=ORANGE).move_to(magnet.get_bottom() + UP*0.1*i) 
            for i in range(5)
        ])
        
        # Etiqueta a la izquierda para no estorbar arriba
        mech_label = Text("Levitador", font_size=20).next_to(magnet, LEFT, buff=0.2)
        
        # Tracker para animación
        y_tracker = ValueTracker(0.015) # 1.5 cm inicial
        
        ball = Circle(radius=0.3, color=Theme.Y, fill_opacity=1)
        ball.add_updater(lambda m: m.next_to(magnet, DOWN, buff=y_tracker.get_value()*100))
        
        mech_group.add(magnet, coil, ball, mech_label)
        
        # --- DERECHA: Gráfica L(y) ---
        # Ajustamos posición para evitar colisiones con el título o la ecuación
        ax = Axes(
            x_range=[0, 0.025, 0.005],
            y_range=[0, 0.06, 0.01],
            x_length=4.0, y_length=3.0, # Más pequeña para evitar problemas
            axis_config={"include_tip": False, "font_size": 18}
        ).to_edge(DR, buff=0.5).shift(LEFT*0.5 + UP*0.5) # Posición segura abajo-derecha
        
        x_label = ax.get_x_axis_label(MathTex("y", color=Theme.Y))
        y_label = ax.get_y_axis_label(MathTex("L(y)", color=WHITE))
        
        # Curva L(y)
        k0, k, a = 0.01, 0.002, 0.005
        func_l = lambda y: k0 + k / (1 + y/a)
        curve = ax.plot(func_l, color=YELLOW)
        
        # Punto móvil
        dot = Dot(color=Theme.Y)
        dot.add_updater(lambda m: m.move_to(ax.c2p(y_tracker.get_value(), func_l(y_tracker.get_value()))))
        
        # Líneas guía
        h_line = always_redraw(lambda: ax.get_horizontal_line(dot.get_left(), line_config={"dashed_ratio": 0.5}))
        v_line = always_redraw(lambda: ax.get_vertical_line(dot.get_bottom(), line_config={"dashed_ratio": 0.5}))
        
        # --- ECUACIÓN: Posición Absoluta ---
        # La colocamos en el cuadrante superior derecho, lejos de todo
        eq_text = MathTex(
            r"L(y) = k_0 + \frac{k}{1 + y/a}",
            font_size=36
        ).move_to(UP*1.5 + RIGHT*3)
        
        eq_text[0][5:7].set_color(Theme.K0)
        eq_text[0][8].set_color(Theme.K)
        eq_text[0][12].set_color(Theme.Y)
        eq_text[0][14].set_color(Theme.A)
        
        # --- ANIMACIÓN ---
        self.add(mech_group, ball)
        self.play(Create(ax), Write(x_label), Write(y_label))
        self.play(Create(curve), Write(eq_text))
        self.play(FadeIn(dot), Create(h_line), Create(v_line))
        
        # Movimiento dinámico
        self.play(y_tracker.animate.set_value(0.002), run_time=2, rate_func=smooth)
        self.play(y_tracker.animate.set_value(0.020), run_time=2, rate_func=smooth)
        self.play(y_tracker.animate.set_value(0.010), run_time=1)
        
        # Texto objetivo debajo de la gráfica
        explicacion = VGroup(
            Text("Objetivo:", color=GREEN, font_size=20),
            Text("Encontrar k0, k, a", font_size=20),
        ).arrange(DOWN).next_to(ax, LEFT, buff=0.2) # Al lado izquierdo de la gráfica para no encimar abajo
        
        self.play(Write(explicacion))
        self.wait(2)


# =============================================================================
# ESCENA 3: OPTIMIZACIÓN (Ajuste de Curvas)
# =============================================================================

class Scene3_Optimizacion(Scene):
    def construct(self):
        _title(self, "Tu Misión: El Ajuste Perfecto", "Minimizar el Error entre Simulación y Realidad")
        
        ax = Axes(
            x_range=[0, 10, 1], y_range=[0, 10, 2],
            x_length=8, y_length=4.5,
            axis_config={"include_tip": False}
        ).shift(DOWN*0.5)
        
        x_lab = Text("Tiempo", font_size=18).next_to(ax.x_axis, DOWN)
        y_lab = Text("Posición y(t)", font_size=18).next_to(ax.y_axis, LEFT).rotate(PI/2)
        
        t = np.linspace(0, 10, 200)
        y_real = 5 + 3*np.sin(t) * np.exp(-0.1*t)
        
        graph_real = ax.plot_line_graph(t, y_real, add_vertex_dots=False, line_color=Theme.REAL)
        label_real = Text("Datos Reales (Target)", font_size=20, color=Theme.REAL).next_to(graph_real, UP, buff=0.1).shift(RIGHT*2)
        
        self.play(Create(ax), Write(x_lab), Write(y_lab))
        self.play(Create(graph_real), Write(label_real))
        
        tracker = ValueTracker(0)
        
        def get_sim_curve():
            progress = tracker.get_value()
            y_bad = 2 + 0.5*t
            y_current = (1-progress)*y_bad + progress*y_real
            return ax.plot_line_graph(t, y_current, add_vertex_dots=False, line_color=Theme.SIM)
            
        graph_sim = always_redraw(get_sim_curve)
        label_sim = Text("Tu Simulación", font_size=20, color=Theme.SIM).next_to(ax.c2p(0, 2), RIGHT)
        
        self.play(Create(graph_sim), Write(label_sim))
        
        err_val = DecimalNumber(100.0, num_decimal_places=2, color=RED, font_size=36)
        err_val.add_updater(lambda d: d.set_value(100 * (1 - tracker.get_value())**2))
        
        err_label = Text("Error (MSE):", font_size=24, color=WHITE).to_corner(UR).shift(DOWN + LEFT)
        err_val.next_to(err_label, RIGHT)
        
        self.play(Write(err_label), Write(err_val))
        
        self.play(tracker.animate.set_value(0.2), run_time=1)
        self.wait(0.5)
        self.play(tracker.animate.set_value(-0.1), run_time=1)
        self.wait(0.5)
        self.play(tracker.animate.set_value(1.0), run_time=3, rate_func=smooth)
        self.play(err_val.animate.set_color(GREEN))
        
        final_text = Text("¡Si logras esto, resolviste el benchmark!", font_size=32, color=YELLOW)
        final_text.move_to(ax.get_center())
        self.play(FadeIn(final_text, scale=0.5))
        self.wait(3)
