#version 450

layout(points) in;
layout (line_strip, max_vertices = 6) out;

layout( set = 0, binding = 0 ) uniform UScene
{
    mat4 M;
    mat4 V;
    mat4 P;
    vec4 camPos;
} uScene;

layout( set = 1, binding = 4 ) uniform sampler2D normalMap;

layout( location = 0 ) in vec2 v2f_tc[];
layout( location = 1 ) in vec4 pos_world[];
layout( location = 2 ) in mat3 tbn[];

layout( location = 0 ) out vec4 line_color;

void main() {
    mat3 TBN = tbn[0];
    // transform N
    vec3 N = normalize(TBN * (texture(normalMap, v2f_tc[0]).rgb * 2.0 - 1.0));

    gl_Position = uScene.P * uScene.V * pos_world[0];
    line_color = vec4(0.0, 1.0, 0.0, 1.0);
    EmitVertex();
    gl_Position = uScene.P * uScene.V * (pos_world[0] + vec4(N * 0.05, 0.0));
    line_color = vec4(0.0, 1.0, 0.0, 1.0);
    EmitVertex();
    EndPrimitive();

}