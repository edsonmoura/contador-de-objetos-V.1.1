[app]
title = Contador de Objetos
package.name = contadorobjetos
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,onnx
version = 0.1.1
requirements = python3,kivy,numpy,opencv,pillow,plyer
orientation = portrait
fullscreen = 0
android.permissions = CAMERA,INTERNET
android.api = 35
android.minapi = 24
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
