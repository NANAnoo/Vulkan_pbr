#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/fwd.hpp"
#include "glm/gtx/euler_angles.hpp"
#include "glm/matrix.hpp"
#include "glm/trigonometric.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <volk/volk.h>

#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <chrono>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "RenderingModel.hpp"
#include "RenderProgram.hpp"
#include "VkUBO.hpp"
#include "VkConstant.hpp"
#include "FpsController.hpp"

namespace
{
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline(s)
		// See sources in cw1/shaders/*. 
#		define SHADERDIR_ "assets/cw2/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "default.frag.spv";
		constexpr char const* kPbrBaseFragShaderPath = SHADERDIR_ "pbr_base.frag.spv";
		constexpr char const* kPbrAMFragShaderPath = SHADERDIR_ "pbr_alpha_mask.frag.spv";
		constexpr char const* kPbrNMFragShaderPath = SHADERDIR_ "pbr_normalmap.frag.spv";
		constexpr char const* kPbrAMNMFragShaderPath = SHADERDIR_ "pbr_am_nm.frag.spv";
		constexpr char const* kNormDebugGeomPath = SHADERDIR_ "normal.geom.spv";
		constexpr char const* kNormDebugFragPath = SHADERDIR_ "normal.frag.spv";
		constexpr char const* kTBNCompVertPath = SHADERDIR_ "tbncomp.vert.spv";
		constexpr char const* kTBNCompFragPath = SHADERDIR_ "tbncomp.frag.spv";
		constexpr char const* kTBNCompGeomPath = SHADERDIR_ "tbn_norm_debug.geom.spv";
#		undef SHADERDIR_

		constexpr char const* kModelPath = "assets/cw2/sponza-pbr.comp5822mesh";

		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 100.f;

		constexpr auto kCameraFov    = 60.0_degf;

		const auto kDepthFormat = VK_FORMAT_D32_SFLOAT;

		enum ShadingMode{
			Basic,
			PBR_base,
			PBR_alpha_mask,
			PBR_alpha_mask_normal_map,
			PBR_TBN_COMPRESS,
		};
	}

	// Local types/structures:
	// Uniform data
	namespace glsl
	{
		struct SceneUniform {
			glm::mat4 M;
			glm::mat4 V;
			glm::mat4 P;
			glm::vec4 camPos;
		};

		struct LightInfo {
			glm::vec4 position;
			glm::vec4 color;
		};
	}

	// Local functions:
	lut::RenderPass create_render_pass( lut::VulkanWindow const& );

	void create_swapchain_framebuffers( 
		lut::VulkanWindow const&, 
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer( 
		lut::VulkanWindow const&,
		lut::Allocator const& 
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		FpsController &aController,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight
	);

	void record_commands(
		VkCommandBuffer aCmdBuff, 
		VkRenderPass aRenderPass, 
		VkFramebuffer aFramebuffer, 
		VkExtent2D const& aImageExtent,
		VkUBO<glsl::SceneUniform> &,
		VkUBO<glsl::LightInfo> &,
		std::function<void(VkCommandBuffer)> const&
	);

	void submit_commands(
		lut::VulkanContext const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);

	void present_results( 
		VkQueue, 
		VkSwapchainKHR, 
		std::uint32_t aImageIndex, 
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);

	// draw helpers
}

#define IS_KEY_DOWN(action) (GLFW_PRESS == action || GLFW_REPEAT == action)

int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	// Configure the GLFW window
	static bool enableMouseNavigation = false;
	static FpsController sController({0, 4, 0}, {0, -90, 0}, 5, 4);
	static float cursor_x = 0, cursor_y = 0, offset_x = 0, offset_y = 0;
	static cfg::ShadingMode sCurrentMode = cfg::Basic;
	static bool shouldGeneratePipeLine = true;
	static bool modeChanged = true;
	static bool shouldDebugNormal = false;
	static bool showDebugNormalMap = false;
	static bool showDebugTBN = false;
	glfwSetKeyCallback( window.window, 
	[]( GLFWwindow* aWindow, int aKey, int, int aAction, int){
		if( GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction )
		{	// close window
			glfwSetWindowShouldClose( aWindow, GLFW_TRUE );
		} // move control: 
		if (aAction == GLFW_RELEASE) {
			sController.onKeyUp(aKey);
			if (aKey == GLFW_KEY_1) {
				// normal mode
				modeChanged = (sCurrentMode != cfg::Basic);
				sCurrentMode = cfg::Basic;
			} else if (aKey == GLFW_KEY_2) {
				// pbr mode
				modeChanged = (sCurrentMode != cfg::PBR_base);
				sCurrentMode = cfg::PBR_base;
			} else if (aKey == GLFW_KEY_3) {
				// pbr alpha mask mode
				modeChanged = (sCurrentMode != cfg::PBR_alpha_mask);
				sCurrentMode = cfg::PBR_alpha_mask;
			} else if (aKey == GLFW_KEY_4) {
				// pbr alpha mask normal map mode
				modeChanged = (sCurrentMode != cfg::PBR_alpha_mask_normal_map);
				sCurrentMode = cfg::PBR_alpha_mask_normal_map;
			} else if (aKey == GLFW_KEY_5) {
				// pbr PBR_TBN_COMPRESS
				modeChanged = (sCurrentMode != cfg::PBR_TBN_COMPRESS);
				sCurrentMode = cfg::PBR_TBN_COMPRESS;
			} else if (aKey == GLFW_KEY_M) {
				showDebugNormalMap = !showDebugNormalMap;
			} else if (aKey == GLFW_KEY_N) {
				shouldDebugNormal = !shouldDebugNormal;
			} else if (aKey == GLFW_KEY_B) {
				showDebugTBN = !showDebugTBN;
			}
		} else if (aAction == GLFW_PRESS) {
			sController.onKeyPress(aKey);
		}
	});

	glfwSetMouseButtonCallback( window.window, 
		[]( GLFWwindow* window, int button, int action, int){
			if( GLFW_MOUSE_BUTTON_RIGHT == button && GLFW_RELEASE == action)
			{
				// change mode
				enableMouseNavigation = !enableMouseNavigation;
				glfwSetInputMode(window, GLFW_CURSOR, enableMouseNavigation ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
			}
		});

	glfwSetCursorPosCallback(window.window, 
		[](GLFWwindow *window, double xpos, double ypos){
			offset_x = float(xpos) - cursor_x;
			offset_y = float(ypos) - cursor_y;
			cursor_x = float(xpos);
			cursor_y = float(ypos);
			if (enableMouseNavigation) {
				sController.onCursorMove(offset_x, offset_y);
			}
		});

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator( window );

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass( window );

	// create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	// a sampler
	lut::Sampler anisotropicSampler = lut::create_default_sampler(window);
	if (window.currentDeviceFeatures.samplerAnisotropy) {
		auto maxAnisotropy = window.maxAnisotropy;
		std::printf("Current device can support anisotropicSampler, maxSamplerAnisotropy is %f", maxAnisotropy);
		anisotropicSampler = lut::create_anisotropic_sampler(window, maxAnisotropy);
	} else {
		std::printf("Current device not support anisotropicSampler! Use defualt sampler");
	}
	 	
	// create all shaders
	lut::ShaderModule baseVert = lut::load_shader_module(window, cfg::kVertShaderPath);
	lut::ShaderModule baseFrag = lut::load_shader_module(window, cfg::kFragShaderPath);
	lut::ShaderModule pbrBaseFrag = lut::load_shader_module(window, cfg::kPbrBaseFragShaderPath);
	lut::ShaderModule pbrAMFrag = lut::load_shader_module(window, cfg::kPbrAMFragShaderPath);
	lut::ShaderModule pbrNMFrag = lut::load_shader_module(window, cfg::kPbrNMFragShaderPath);
	lut::ShaderModule pbrAMNMFrag = lut::load_shader_module(window, cfg::kPbrAMNMFragShaderPath);
	lut::ShaderModule normDebugGeom = lut::load_shader_module(window, cfg::kNormDebugGeomPath);
	lut::ShaderModule normDebugFrag = lut::load_shader_module(window, cfg::kNormDebugFragPath);
	lut::ShaderModule tbnCompVert = lut::load_shader_module(window, cfg::kTBNCompVertPath);
	lut::ShaderModule tbnCompFrag = lut::load_shader_module(window, cfg::kTBNCompFragPath);
	lut::ShaderModule tbnCompGeom = lut::load_shader_module(window, cfg::kTBNCompGeomPath);

	// create scene ubo
	VkUBO<glsl::SceneUniform> sceneUBO(window, allocator, dpool, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_GEOMETRY_BIT, 0);
	sceneUBO.data = std::make_unique<glsl::SceneUniform>();
	
	// create light ubo
	VkUBO<glsl::LightInfo> lightUBO(window, allocator, dpool, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	lightUBO.data = std::make_unique<glsl::LightInfo>();
	lightUBO.data->position = glm::vec4(0, 2, 0, 1);
	lightUBO.data->color = glm::vec4(1, 1, 1, 1);

	// create constant
	struct FrameSize {
		float width;
		float height;
	};
	// used in GEOMETRY shader
	VkConstant<FrameSize> frameSizeConst(0, VK_SHADER_STAGE_GEOMETRY_BIT);

	// pipe generator, set up basic informations
	PipeLineGenerator basicPipeGen;
	basicPipeGen
	.addDescLayout(sceneUBO.layout.handle)
	.addDescLayout(lightUBO.layout.handle)
	.enableBlend(false)
	.setCullMode(VK_CULL_MODE_BACK_BIT)
	.setPolyGonMode(VK_POLYGON_MODE_FILL)
	.enableDepthTest(true)
	.setRenderMode(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	PipeLineGenerator normMapDebugPipeGen;
	normMapDebugPipeGen
	.addDescLayout(sceneUBO.layout.handle)
	.enableBlend(false)
	.setCullMode(VK_CULL_MODE_BACK_BIT)
	.setPolyGonMode(VK_POLYGON_MODE_FILL)
	.enableDepthTest(true)
	.setRenderMode(VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
	.bindVertShader(baseVert)
	.bindGeomShader(normDebugGeom)
	.bindFragShader(normDebugFrag)
	.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));

	PipeLineGenerator tbnCompPipeGen;
	tbnCompPipeGen
	.addDescLayout(sceneUBO.layout.handle)
	.enableBlend(false)
	.setCullMode(VK_CULL_MODE_BACK_BIT)
	.setPolyGonMode(VK_POLYGON_MODE_FILL)
	.enableDepthTest(true)
	.setRenderMode(VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
	.bindVertShader(tbnCompVert)
	.bindGeomShader(tbnCompGeom)
	.bindFragShader(normDebugFrag)
	.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));

	// cache all pipeline
	std::unordered_map<cfg::ShadingMode, std::vector<RenderPipeLine>> pipelineMap;

	// create frambuffer and depth buffer
	auto [depthBuffer, depthBufferView] = create_depth_buffer( window, allocator );

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers( window, renderPass.handle, framebuffers, depthBufferView.handle);

	lut::CommandPool cpool = lut::create_command_pool( window, 
														VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | 
														VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT );
	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;
	
	for( std::size_t i = 0; i < framebuffers.size(); ++i )
	{
		cbuffers.emplace_back( lut::alloc_command_buffer( window, cpool.handle ) );
		cbfences.emplace_back( lut::create_fence( window, VK_FENCE_CREATE_SIGNALED_BIT ) );
	}

	lut::Semaphore imageAvailable = lut::create_semaphore( window );
	lut::Semaphore renderFinished = lut::create_semaphore( window );

	// Load data
	std::shared_ptr<RenderingModel> model = std::make_shared<RenderingModel>();
	{
		auto tmp = load_baked_model(cfg::kModelPath);
		model->load(window, allocator, tmp);
		model->createSetsWith(window, dpool.handle, anisotropicSampler.handle);
		// upload data
		RenderingModel::uploadScope(
			window, 
			[&model](VkCommandBuffer cmd){
				model->upload(cmd);
			}
		);
	}
	normMapDebugPipeGen = model->bindPipeLine(normMapDebugPipeGen);
	normMapDebugPipeGen.addDescLayout(model->debugLayout.handle);
	auto normMapDebugPipe = normMapDebugPipeGen.generate(window, renderPass.handle);
	
	tbnCompPipeGen = model->bindTbnCompressedPipeLine(tbnCompPipeGen);
	tbnCompPipeGen.addDescLayout(model->pbrFullLayout.handle);
	auto tbnCompPipe = tbnCompPipeGen.generate(window, renderPass.handle);

	// Application main loop
	bool recreateSwapchain = false;
	auto previous = std::chrono::system_clock::now();
	while (!glfwWindowShouldClose(window.window)) {
		// Let GLFW process events.
		glfwPollEvents(); // or: glfwWaitEvents()
		auto now = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - previous).count() / 1000000000.f;
		previous = now;
		// update control info
		sController.update(duration);
		// update uniforms
		update_scene_uniforms(
			*sceneUBO.data.get(), 
			sController,
			window.swapchainExtent.width, 
			window.swapchainExtent.height
		);
		// Recreate swap chain
		if (recreateSwapchain)
		{
			//We need to destroy several objects, which may still be in use by the GPU
			vkDeviceWaitIdle( window.device );

			// Recreate them
			auto const changes = recreate_swapchain( window );

			if( changes.changedFormat ) {
				renderPass = create_render_pass( window );
			}
			if( changes.changedSize ) {
				std::tie(depthBuffer,depthBufferView) = create_depth_buffer( window, allocator );
				basicPipeGen.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));
				normMapDebugPipeGen.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));
				tbnCompPipeGen.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));
			}

			framebuffers.clear();
			create_swapchain_framebuffers( window, renderPass.handle, framebuffers, depthBufferView.handle); 
			
			recreateSwapchain = false;
			shouldGeneratePipeLine = true;
			// clear all pipelines
			pipelineMap.clear();
			continue;
		}
		
		// generate pipeline
		if (shouldGeneratePipeLine || modeChanged) {
			basicPipeGen.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));
			
			// check if there is a pipeline cache
			// also lazy loading here
			if (shouldGeneratePipeLine) {
				normMapDebugPipeGen.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));
				tbnCompPipeGen.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));
				normMapDebugPipe = normMapDebugPipeGen.generate(window, renderPass.handle);
				tbnCompPipe = tbnCompPipeGen.generate(window, renderPass.handle);
			}
			
			if (sCurrentMode == cfg::Basic) {
				pipelineMap[cfg::Basic].emplace_back(
					model->
					bindPipeLine(basicPipeGen)
					.bindVertShader(baseVert)
					.bindFragShader(baseFrag)
					.generate(window, renderPass.handle)
				);
			} else if (sCurrentMode == cfg::PBR_base) {
				pipelineMap[cfg::PBR_base].emplace_back(
					model->
					bindPipeLine(basicPipeGen)
					.bindVertShader(baseVert)
					.bindFragShader(pbrBaseFrag)
					.addDescLayout(model->pbrBaseLayout.handle)
					.generate(window, renderPass.handle)
				);
			} else if (sCurrentMode == cfg::PBR_alpha_mask) {
				pipelineMap[cfg::PBR_alpha_mask].emplace_back(
					model->
					bindPipeLine(basicPipeGen)
					.bindVertShader(baseVert)
					.bindFragShader(pbrBaseFrag)
					.addDescLayout(model->pbrBaseLayout.handle)
					.generate(window, renderPass.handle)
				);
				// only alpa mask
				pipelineMap[cfg::PBR_alpha_mask].emplace_back(
					model->
					bindPipeLine(basicPipeGen)
					.bindVertShader(baseVert)
					.bindFragShader(pbrAMFrag)
					.addDescLayout(model->pbrHalfLayout.handle)
					.generate(window, renderPass.handle)
				);
			} else if (sCurrentMode == cfg::PBR_alpha_mask_normal_map) {
				pipelineMap[cfg::PBR_alpha_mask_normal_map].emplace_back(
					model->
					bindPipeLine(basicPipeGen)
					.bindVertShader(baseVert)
					.bindFragShader(pbrBaseFrag)
					.addDescLayout(model->pbrBaseLayout.handle)
					.generate(window, renderPass.handle)
				);
				// only alpa mask
				pipelineMap[cfg::PBR_alpha_mask_normal_map].emplace_back(
					model->
					bindPipeLine(basicPipeGen)
					.bindVertShader(baseVert)
					.bindFragShader(pbrAMFrag)
					.addDescLayout(model->pbrHalfLayout.handle)
					.generate(window, renderPass.handle)
				);
				// only normal map
				pipelineMap[cfg::PBR_alpha_mask_normal_map].emplace_back(
					model->
					bindPipeLine(basicPipeGen)
					.bindVertShader(baseVert)
					.bindFragShader(pbrNMFrag)
					.addDescLayout(model->pbrHalfLayout.handle)
					.generate(window, renderPass.handle)
				);
				// all
				pipelineMap[cfg::PBR_alpha_mask_normal_map].emplace_back(
					model->
					bindPipeLine(basicPipeGen)
					.bindVertShader(baseVert)
					.bindFragShader(pbrAMNMFrag)
					.addDescLayout(model->pbrFullLayout.handle)
					.generate(window, renderPass.handle)
				);
			} else if (sCurrentMode == cfg::PBR_TBN_COMPRESS) {
				pipelineMap[cfg::PBR_TBN_COMPRESS].emplace_back(
					model->
					bindTbnCompressedPipeLine(basicPipeGen)
					.bindVertShader(tbnCompVert)
					.bindFragShader(tbnCompFrag)
					.addDescLayout(model->pbrFullLayout.handle)
					.generate(window, renderPass.handle)
				);
			}

			shouldGeneratePipeLine = false;
			modeChanged = false;
		}

		// acquire swapchain image.
		std::uint32_t imageIndex = 0;
		const auto acquireRes = vkAcquireNextImageKHR(
			window.device, 
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);
		if( VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes ) {
			recreateSwapchain = true;
			continue;
		}
		if (VK_SUCCESS != acquireRes) {
			throw lut::Error("Unable to acquire next swapchain image\n"
				"vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}
		// wait for command buffer to be available
		// make sure that the command buffer is no longer in use
		assert( std::size_t(imageIndex) < cbfences.size());
		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, 
			std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res) {
				throw lut::Error( "Unable to wait for command buffer fence %u\n"
					"vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}
		if( auto const res = vkResetFences( window.device, 1, &cbfences[imageIndex].handle ); 
			VK_SUCCESS != res ) {
				throw lut::Error( "Unable to reset command buffer fence %u\n"
					"vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		// record and submit commands
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		auto BeginPipeline = [&](VkCommandBuffer cmdBuffer, const RenderPipeLine &pipe) {
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe.handle);
			vkCmdBindDescriptorSets(cmdBuffer, 
				VK_PIPELINE_BIND_POINT_GRAPHICS, 
				pipe.layout.handle, 0,
				1, 
				&sceneUBO.set, 
				0, nullptr
			);
			vkCmdBindDescriptorSets(cmdBuffer, 
				VK_PIPELINE_BIND_POINT_GRAPHICS, 
				pipe.layout.handle, 1,
				1, 
				&lightUBO.set, 
				0, nullptr
			);
		};

		auto BindingMatSet = [](const RenderPipeLine &pipe, VkCommandBuffer cmdBuffer, VkDescriptorSet mat_set) {
			vkCmdBindDescriptorSets(cmdBuffer, 
				VK_PIPELINE_BIND_POINT_GRAPHICS, 
				pipe.layout.handle, 2,
				1, 
				&mat_set, 
				0, nullptr
			);
		};

		record_commands(
			cbuffers[imageIndex], 
			renderPass.handle, 
			framebuffers[imageIndex].handle,
			window.swapchainExtent,
			sceneUBO,
			lightUBO,
			// real draw task
			[&](VkCommandBuffer cmdBuffer) {
				if (sCurrentMode == cfg::Basic) {
					auto& pipe = pipelineMap[cfg::Basic][0];
					BeginPipeline(cmdBuffer, pipe);
					model->onDraw(RenderingModel::Base, RenderingModel::Normal, cmdBuffer, [&](VkDescriptorSet mat_set) {});
					model->onDraw(RenderingModel::AlphaMasked, RenderingModel::Normal, cmdBuffer, [&](VkDescriptorSet mat_set) {});
					model->onDraw(RenderingModel::NormalMapped, RenderingModel::Normal, cmdBuffer, [&](VkDescriptorSet mat_set) {});
					model->onDraw(RenderingModel::All, RenderingModel::Normal, cmdBuffer, [&](VkDescriptorSet mat_set) {});
				} else if (sCurrentMode == cfg::PBR_base) {
					auto& pipe = pipelineMap[cfg::PBR_base][0];
					BeginPipeline(cmdBuffer, pipe);
					model->onDraw(RenderingModel::Base, RenderingModel::Normal, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe, cmdBuffer, mat_set);
					});
					model->onDraw(RenderingModel::NormalMapped, RenderingModel::Normal, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe, cmdBuffer, mat_set);
					});
					model->onDraw(RenderingModel::AlphaMasked, RenderingModel::Normal, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe, cmdBuffer, mat_set);
					});
					model->onDraw(RenderingModel::All, RenderingModel::Normal, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe, cmdBuffer, mat_set);
					});
				} else if (sCurrentMode == cfg::PBR_alpha_mask) {
					auto& pipe = pipelineMap[cfg::PBR_alpha_mask][0];
					auto& pipe_alpha_test = pipelineMap[cfg::PBR_alpha_mask][1];
					BeginPipeline(cmdBuffer, pipe);
					model->onDraw(RenderingModel::Base, RenderingModel::AlphaMask, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe, cmdBuffer, mat_set);
					});
					model->onDraw(RenderingModel::NormalMapped, RenderingModel::AlphaMask, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe, cmdBuffer, mat_set);
					});

					BeginPipeline(cmdBuffer, pipe_alpha_test);
					model->onDraw(RenderingModel::AlphaMasked, RenderingModel::AlphaMask, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe_alpha_test, cmdBuffer, mat_set);
					});
					model->onDraw(RenderingModel::All, RenderingModel::AlphaMask, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe_alpha_test, cmdBuffer, mat_set);
					});
				} else if (sCurrentMode == cfg::PBR_alpha_mask_normal_map) {
					auto& pipe = pipelineMap[cfg::PBR_alpha_mask_normal_map][0];
					auto& pipe_alpha_test = pipelineMap[cfg::PBR_alpha_mask_normal_map][1];
					auto& pipe_normal_map = pipelineMap[cfg::PBR_alpha_mask_normal_map][2];
					auto& pipe_normal_map_alpha_test = pipelineMap[cfg::PBR_alpha_mask_normal_map][3];
					BeginPipeline(cmdBuffer, pipe);
					model->onDraw(RenderingModel::Base, RenderingModel::Full, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe, cmdBuffer, mat_set);
					});
					BeginPipeline(cmdBuffer, pipe_alpha_test);
					model->onDraw(RenderingModel::AlphaMasked, RenderingModel::Full, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe_alpha_test, cmdBuffer, mat_set);
					});
					BeginPipeline(cmdBuffer, pipe_normal_map);
					model->onDraw(RenderingModel::NormalMapped, RenderingModel::Full, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe_normal_map, cmdBuffer, mat_set);
					});
					BeginPipeline(cmdBuffer, pipe_normal_map_alpha_test);
					model->onDraw(RenderingModel::All, RenderingModel::Full, cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe_normal_map_alpha_test, cmdBuffer, mat_set);
					});
				} else if (sCurrentMode == cfg::PBR_TBN_COMPRESS) {
					auto &pipe = pipelineMap[cfg::PBR_TBN_COMPRESS][0];
					BeginPipeline(cmdBuffer, pipe);
					model->tbnCompressedDraw(cmdBuffer, [&](VkDescriptorSet mat_set) {
						BindingMatSet(pipe, cmdBuffer, mat_set);
					});
				}
				if (showDebugNormalMap) {
					vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, normMapDebugPipe.pipe.handle);
					vkCmdBindDescriptorSets(cmdBuffer, 
						VK_PIPELINE_BIND_POINT_GRAPHICS, 
						normMapDebugPipe.layout.handle, 0,
						1, 
						&sceneUBO.set, 
						0, nullptr
					);
					model->onDraw(RenderingModel::NormalMapped, RenderingModel::NormalDebug, cmdBuffer, [&](VkDescriptorSet mat_set) {
						vkCmdBindDescriptorSets(cmdBuffer, 
							VK_PIPELINE_BIND_POINT_GRAPHICS, 
							normMapDebugPipe.layout.handle, 1,
							1, 
							&mat_set, 
							0, nullptr
						);
					});
					model->onDraw(RenderingModel::All, RenderingModel::NormalDebug, cmdBuffer, [&](VkDescriptorSet mat_set) {
						vkCmdBindDescriptorSets(cmdBuffer, 
							VK_PIPELINE_BIND_POINT_GRAPHICS, 
							normMapDebugPipe.layout.handle, 1,
							1, 
							&mat_set, 
							0, nullptr
						);
					});
				}
				if (showDebugTBN) {
					vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, tbnCompPipe.pipe.handle);
					vkCmdBindDescriptorSets(cmdBuffer, 
						VK_PIPELINE_BIND_POINT_GRAPHICS, 
						tbnCompPipe.layout.handle, 0,
						1, 
						&sceneUBO.set, 
						0, nullptr
					);
					model->tbnCompressedDraw(cmdBuffer, [&](const VkDescriptorSet& mat_set) {
						vkCmdBindDescriptorSets(cmdBuffer, 
							VK_PIPELINE_BIND_POINT_GRAPHICS, 
							tbnCompPipe.layout.handle, 1,
							1, 
							&mat_set, 
							0, nullptr
						);
					});
				}
			}
		);

		submit_commands(window, 
			cbuffers[imageIndex],
			cbfences[imageIndex].handle, 
			imageAvailable.handle, 
			renderFinished.handle
		);

		// present rendered images.
		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);
		std::printf("\rFPS: %.3f", 1.f / duration);
	}
	
	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle( window.device );

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
	void update_scene_uniforms(
		glsl::SceneUniform& aSceneUniforms, 
		FpsController& aController,
		std::uint32_t aFramebufferWidth, 
		std::uint32_t aFramebufferHeight )
	{
		
		// initialize SceneUniform members
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);
		aSceneUniforms.P = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
	
		aSceneUniforms.P[1][1] *= -1.f; // mirror Y axis

		aSceneUniforms.camPos = glm::vec4(aController.m_position, 1.f);
		
		// move camera to center
		glm::mat4 V = glm::translate(-aController.m_position);
		// ratate the camera back to the front direction
		V = glm::eulerAngleXYZ(
			glm::radians(-aController.m_rotation.x), 
			glm::radians(-aController.m_rotation.y), 0.f) * V;
		
		// V is just move object to camera space
		aSceneUniforms.V = V;
		// no model change here
		aSceneUniforms.M = glm::mat4(1.f);
	}
}

namespace
{
	lut::RenderPass create_render_pass( lut::VulkanWindow const& aWindow )
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; 
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; 
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{}; 
		depthAttachment.attachment = 1; // this refers to attachments[1]
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		
		VkSubpassDescription subpasses[1]{}; 
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1; 
		subpasses[0].pColorAttachments = subpassAttachments; 
		subpasses[0].pDepthStencilAttachment = &depthAttachment; // New!

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2; 
		passInfo.pAttachments = attachments; 
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0;
		passInfo.pDependencies = nullptr;

		VkRenderPass rpass = VK_NULL_HANDLE; 
		if( auto const res = vkCreateRenderPass( aWindow.device, &passInfo, nullptr, &rpass); 
			VK_SUCCESS != res ) {
			throw lut::Error( "Unable to create render pass\n" 
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}
		return lut::RenderPass( aWindow.device, rpass );
	}

	void create_swapchain_framebuffers( 
		lut::VulkanWindow const& aWindow, 
		VkRenderPass aRenderPass, 
		std::vector<lut::Framebuffer>& aFramebuffers,
		VkImageView aDepthView)
	{
		assert( aFramebuffers.empty() );

		for (std::uint32_t i = 0; i < aWindow.swapViews.size(); i ++) {
			VkImageView attachments[2] = {
				aWindow.swapViews[i],
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; 
			fbInfo.renderPass = aRenderPass; 
			fbInfo.attachmentCount = 2; 
			fbInfo.pAttachments = attachments; 
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1; 

			VkFramebuffer fb = VK_NULL_HANDLE; 
			if( auto const res = vkCreateFramebuffer( aWindow.device, &fbInfo, nullptr, &fb); 
				VK_SUCCESS != res ) {
				throw lut::Error( "Unable to create framebuffer for swap chain image %zu\n"
					"vkCreateFramebuffer() returned %s", i,lut::to_string(res).c_str());
			}
			aFramebuffers.emplace_back( lut::Framebuffer( aWindow.device, fb ) );
		}

		assert( aWindow.swapViews.size() == aFramebuffers.size() );
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer( 
		lut::VulkanWindow const& aWindow,
		lut::Allocator const& aAllocator) 
	{ 
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D; 
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1; 
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT; 
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; 
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE; 
		if( auto const res = vmaCreateImage( aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr ); 
			VK_SUCCESS != res )
		{
			throw lut::Error( "Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage( aAllocator.allocator, image, allocation );

		// Create the image view
		VkImageViewCreateInfo viewInfo{}; 
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image; 
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat; 
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1, 0, 1};
		
		VkImageView view = VK_NULL_HANDLE;

		if( auto const res = vkCreateImageView( aWindow.device, &viewInfo, nullptr, &view );
			VK_SUCCESS != res )
		{
			throw lut::Error( "Unable to create image view\n" 
				"vkCreateImageView() returned %s", lut::to_string(res).c_str() );
		}
		return { std::move(depthImage), lut::ImageView( aWindow.device, view ) };
	}

	void record_commands(
		VkCommandBuffer aCmdBuff, 
		VkRenderPass aRenderPass, 
		VkFramebuffer aFramebuffer, 
		VkExtent2D const& aImageExtent,
		VkUBO<glsl::SceneUniform> &sceneUBO,
		VkUBO<glsl::LightInfo> &lightUBO,
		std::function<void(VkCommandBuffer)> const& renderTask)
	{
		// Begin recording commands
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;
		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &beginInfo); VK_SUCCESS != res) {
			throw lut::Error("Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// update uniforms
		sceneUBO.upload(aCmdBuff, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
		lightUBO.upload(aCmdBuff, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		// Begin render pass
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f; // Clear to a dark gray background.
		clearValues[0].color.float32[1] = 0.1f; // Clear to a dark gray background.
		clearValues[0].color.float32[2] = 0.1f; // Clear to a dark gray background.
		clearValues[0].color.float32[3] = 1.f; // Clear to a dark gray background.

		clearValues[1].depthStencil.depth = 1.f; // clear depth as 1.0f

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = aImageExtent;
		passInfo.clearValueCount = 2;  
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
		
		renderTask(aCmdBuff);

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);

		// End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res) {
			throw lut::Error("Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
	}

	void submit_commands( lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore )
	{
		VkPipelineStageFlags waitPipelineStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;


		if( auto const res = vkQueueSubmit( aContext.graphicsQueue, 1, &submitInfo, aFence )
			; VK_SUCCESS != res ) {
			throw lut::Error( "Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	void present_results( VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain )
	{
		// present the results
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1; 
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1; 
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR( aPresentQueue, &presentInfo );

		if( VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes ) {
			aNeedToRecreateSwapchain = true; 
		}
		else if( VK_SUCCESS != presentRes ) {
			throw lut::Error( "Unable present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str());
		}
	}

	// helper
}


//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
