"""
Video explicativo: Observador KAN-PINN para Levitador Magnético
Autor: José de Jesús Santana Ramírez
ORCID: 0000-0002-6183-7379

Ejecutar:
    manim-slides render kan_pinn_explicacion.py KANPINNExplicacion -ql
    manim-slides convert KANPINNExplicacion kan_pinn.html --one-file
"""

from manim import *
from manim_slides import Slide

class KANPINNExplicacion(Slide):
    def construct(self):
        self.camera.background_color = WHITE
        
        # =====================================================================
        # Slide 1: Título
        # =====================================================================
        titulo = Text("Observador KAN-PINN", font_size=48, color=BLACK)
        subtitulo = Text("Levitador Magnético", font_size=32, color=GRAY)
        autor = Text("José de Jesús Santana Ramírez", font_size=24, color=DARK_GRAY)
        orcid = Text("ORCID: 0000-0002-6183-7379", font_size=18, color=BLUE)
        
        titulo_group = VGroup(titulo, subtitulo, autor, orcid).arrange(DOWN, buff=0.4)
        
        self.play(Write(titulo), run_time=1)
        self.play(FadeIn(subtitulo), run_time=0.5)
        self.play(FadeIn(autor), FadeIn(orcid), run_time=0.5)
        self.next_slide()
        
        self.play(FadeOut(titulo_group))
        
        # =====================================================================
        # Slide 2: El Problema
        # =====================================================================
        prob_titulo = Text("El Problema", font_size=36, color=BLACK).to_edge(UP)
        
        problema_texto = VGroup(
            Text("Sin sensor de posición:", font_size=24, color=BLACK),
            Text("Solo medimos corriente (i) y voltaje (u)", font_size=20, color=DARK_GRAY),
            Text("", font_size=12),
            Text("Objetivo:", font_size=24, color=BLACK),
            Text("Estimar posición (y) usando física + IA", font_size=20, color=DARK_GRAY),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).shift(UP*0.5)
        
        # Diagrama simple
        bobina = Circle(radius=0.8, color=BLUE, fill_opacity=0.3).shift(DOWN*1.5 + LEFT*2)
        esfera = Circle(radius=0.3, color=GRAY, fill_opacity=0.8).shift(DOWN*1.5 + LEFT*2 + DOWN*1.5)
        pregunta = Text("y = ?", font_size=28, color=RED).next_to(esfera, RIGHT)
        
        flecha_i = Arrow(start=LEFT*4, end=bobina.get_left(), color=GREEN)
        label_i = Text("i", font_size=24, color=GREEN).next_to(flecha_i, UP)
        
        self.play(Write(prob_titulo))
        self.play(FadeIn(problema_texto))
        self.play(Create(bobina), Create(esfera), Write(pregunta))
        self.play(Create(flecha_i), Write(label_i))
        self.next_slide()
        
        self.play(FadeOut(VGroup(prob_titulo, problema_texto, bobina, esfera, pregunta, flecha_i, label_i)))
        
        # =====================================================================
        # Slide 3: Física del Sistema
        # =====================================================================
        fis_titulo = Text("Física del Sistema", font_size=36, color=BLACK).to_edge(UP)
        
        eq_inductancia = MathTex(r"L(y) = k_0 + \frac{k}{1 + y/a}", font_size=36, color=BLACK)
        eq_flujo = MathTex(r"\phi = L(y) \cdot i", font_size=36, color=BLACK)
        eq_voltaje = MathTex(r"\phi = \int (u - R \cdot i) \, dt", font_size=36, color=BLACK)
        
        ecuaciones = VGroup(eq_inductancia, eq_flujo, eq_voltaje).arrange(DOWN, buff=0.6)
        
        # Parámetros
        params = VGroup(
            MathTex(r"k_0 = 0.0704 \text{ H}", font_size=24, color=DARK_GRAY),
            MathTex(r"k = 0.0327 \text{ H}", font_size=24, color=DARK_GRAY),
            MathTex(r"a = 0.0052 \text{ m}", font_size=24, color=DARK_GRAY),
            MathTex(r"R = 2.72 \, \Omega", font_size=24, color=DARK_GRAY),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).to_edge(RIGHT).shift(DOWN*0.5)
        
        self.play(Write(fis_titulo))
        self.play(Write(eq_inductancia))
        self.next_slide()
        self.play(Write(eq_flujo))
        self.play(Write(eq_voltaje))
        self.play(FadeIn(params))
        self.next_slide()
        
        self.play(FadeOut(VGroup(fis_titulo, ecuaciones, params)))
        
        # =====================================================================
        # Slide 4: Arquitectura KAN
        # =====================================================================
        kan_titulo = Text("Arquitectura KAN", font_size=36, color=BLACK).to_edge(UP)
        
        # Capas como rectángulos
        capa1 = Rectangle(width=2, height=1, color=BLUE, fill_opacity=0.3)
        capa2 = Rectangle(width=2, height=1, color=BLUE, fill_opacity=0.3)
        capa3 = Rectangle(width=2, height=1, color=BLUE, fill_opacity=0.3)
        
        capas = VGroup(capa1, capa2, capa3).arrange(RIGHT, buff=1.5)
        
        # Labels
        l1 = Text("3 → 32", font_size=18, color=BLACK).move_to(capa1)
        l2 = Text("32 → 32", font_size=18, color=BLACK).move_to(capa2)
        l3 = Text("32 → 1", font_size=18, color=BLACK).move_to(capa3)
        
        # Flechas
        arr1 = Arrow(capa1.get_right(), capa2.get_left(), color=DARK_GRAY, buff=0.1)
        arr2 = Arrow(capa2.get_right(), capa3.get_left(), color=DARK_GRAY, buff=0.1)
        
        # Entrada y salida
        entrada = VGroup(
            Text("i", font_size=20, color=GREEN),
            Text("L", font_size=20, color=ORANGE),
            Text("u", font_size=20, color=PURPLE),
        ).arrange(DOWN, buff=0.2).next_to(capa1, LEFT, buff=0.5)
        
        salida = Text("y", font_size=24, color=RED).next_to(capa3, RIGHT, buff=0.5)
        
        arr_in = Arrow(entrada.get_right(), capa1.get_left(), color=DARK_GRAY, buff=0.1)
        arr_out = Arrow(capa3.get_right(), salida.get_left(), color=DARK_GRAY, buff=0.1)
        
        # B-spline nota
        bspline = Text("B-splines + Residual", font_size=18, color=GRAY).next_to(capas, DOWN, buff=0.5)
        
        kan_group = VGroup(capas, l1, l2, l3, arr1, arr2, entrada, salida, arr_in, arr_out, bspline)
        
        self.play(Write(kan_titulo))
        self.play(Create(capa1), Create(capa2), Create(capa3))
        self.play(Write(l1), Write(l2), Write(l3))
        self.play(Create(arr1), Create(arr2))
        self.play(FadeIn(entrada), FadeIn(salida), Create(arr_in), Create(arr_out))
        self.play(Write(bspline))
        self.next_slide()
        
        self.play(FadeOut(VGroup(kan_titulo, kan_group)))
        
        # =====================================================================
        # Slide 5: Pérdida PINN
        # =====================================================================
        pinn_titulo = Text("Pérdida Physics-Informed", font_size=36, color=BLACK).to_edge(UP)
        
        loss_total = MathTex(r"\mathcal{L} = \mathcal{L}_{datos} + \lambda \cdot \mathcal{L}_{fisica}", 
                             font_size=36, color=BLACK)
        
        loss_datos = MathTex(r"\mathcal{L}_{datos} = \text{MSE}(y_{pred}, y_{sensor})", 
                             font_size=28, color=DARK_GRAY)
        
        loss_fisica = MathTex(r"\mathcal{L}_{fisica} = \text{MSE}(L_{pred}, L_{modelo})", 
                              font_size=28, color=DARK_GRAY)
        
        losses = VGroup(loss_total, loss_datos, loss_fisica).arrange(DOWN, buff=0.8)
        
        self.play(Write(pinn_titulo))
        self.play(Write(loss_total))
        self.next_slide()
        self.play(Write(loss_datos))
        self.play(Write(loss_fisica))
        self.next_slide()
        
        self.play(FadeOut(VGroup(pinn_titulo, losses)))
        
        # =====================================================================
        # Slide 6: Resultados
        # =====================================================================
        res_titulo = Text("Resultados", font_size=36, color=BLACK).to_edge(UP)
        
        tabla = VGroup(
            Text("Métrica          Valor", font_size=24, color=BLACK),
            Text("─────────────────────", font_size=24, color=GRAY),
            Text("Correlación      0.589", font_size=22, color=DARK_GRAY),
            Text("MAE              2.88 mm", font_size=22, color=DARK_GRAY),
            Text("Datasets         5 (~13k)", font_size=22, color=DARK_GRAY),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        
        self.play(Write(res_titulo))
        self.play(FadeIn(tabla))
        self.next_slide()
        
        self.play(FadeOut(VGroup(res_titulo, tabla)))
        
        # =====================================================================
        # Slide 7: Validación con Metaheurísticos
        # =====================================================================
        val_titulo = Text("Validación Cruzada", font_size=36, color=BLACK).to_edge(UP)
        
        ciclo = VGroup(
            Text("1. Metaheurísticos → k₀, k, a", font_size=22, color=BLACK),
            Text("2. KAN-PINN entrenado con física", font_size=22, color=BLACK),
            Text("3. Comparar estimación vs sensor", font_size=22, color=BLACK),
            Text("4. Validar modelo físico", font_size=22, color=BLACK),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        
        check = Text("✓ Validación mutua", font_size=28, color=GREEN).next_to(ciclo, DOWN, buff=0.8)
        
        self.play(Write(val_titulo))
        for item in ciclo:
            self.play(FadeIn(item), run_time=0.5)
        self.next_slide()
        self.play(Write(check))
        self.next_slide()
        
        self.play(FadeOut(VGroup(val_titulo, ciclo, check)))
        
        # =====================================================================
        # Slide 8: Fin
        # =====================================================================
        fin = Text("¡Gracias!", font_size=48, color=BLACK)
        contacto = VGroup(
            Text("José de Jesús Santana Ramírez", font_size=24, color=DARK_GRAY),
            Text("ORCID: 0000-0002-6183-7379", font_size=20, color=BLUE),
            Text("jesus.santana@uaq.mx", font_size=20, color=DARK_GRAY),
        ).arrange(DOWN, buff=0.3).next_to(fin, DOWN, buff=1)
        
        self.play(Write(fin))
        self.play(FadeIn(contacto))
        self.next_slide()
