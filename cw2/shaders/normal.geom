#version 450

layout(points) in;
layout (line_strip, max_vertices = 2) out;

layout( set = 0, binding = 0 ) uniform UScene
{
    mat4 M;
    mat4 V;
    mat4 P;
    vec4 camPos;
} uScene;

layout( location = 0 ) in vec2 v2f_tc[];
layout( location = 1 ) in vec4 pos_world[];
layout( location = 2 ) in vec3 normal_world[];
layout( location = 3 ) in mat3 tbn[];

layout( location = 0 ) out vec4 line_color;

void main() {
    // transform N
    vec3 N = normal_world[0];

    gl_Position = uScene.P * uScene.V * pos_world[0];
    line_color = vec4(1.0, 0.0, 0.0, 1.0);
    EmitVertex();
    gl_Position = uScene.P * uScene.V * (pos_world[0] + vec4(N * 0.05, 0.0));
    line_color = vec4(1.0, 0.0, 0.0, 1.0);
    EmitVertex();
    EndPrimitive();

}