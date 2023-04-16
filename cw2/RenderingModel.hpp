#pragma once

#include "RenderProgram.hpp"
#include "baked_model.hpp"
#include "VkVertex.hpp"
#include <functional>
#include <memory>

namespace {
    struct RenderingMesh {
        std::unique_ptr<VkVBO> positions = nullptr;
        std::unique_ptr<VkVBO> normals = nullptr;
        std::unique_ptr<VkVBO> texcoords = nullptr;

        std::unique_ptr<VkIBO> indices = nullptr;

        RenderingMesh() noexcept = default;
        // disable copy
        RenderingMesh(const RenderingMesh&) = delete;
        RenderingMesh operator=(const RenderingMesh&) = delete;

        // enable move
        RenderingMesh(RenderingMesh&& other) noexcept {
            positions = std::move(other.positions);
            normals = std::move(other.normals);
            texcoords = std::move(other.texcoords);
            indices = std::move(other.indices);
        }
        RenderingMesh& operator=(RenderingMesh&& other) noexcept {
            positions = std::move(other.positions);
            normals = std::move(other.normals);
            texcoords = std::move(other.texcoords);
            indices = std::move(other.indices);
            return *this;
        }
    };

    class RenderingModel {
    public:
        RenderingModel() noexcept = default;
        // disable copy
        RenderingModel(const RenderingModel&) = delete;
        RenderingModel operator=(const RenderingModel&) = delete;

        // enable move
        RenderingModel(RenderingModel&& other) noexcept {
            meshes = std::move(other.meshes);
        }
        RenderingModel& operator=(RenderingModel&& other) noexcept {
            meshes = std::move(other.meshes);
            return *this;
        }

        void load(
            lut::VulkanContext const& aContext, 
            lut::Allocator const& aAllocator,
            const BakedModel& model) 
        {
            meshes.resize(model.meshes.size());
            for (size_t i = 0; i < model.meshes.size(); ++i) {
                auto& mesh = model.meshes[i];
                auto& renderingMesh = meshes[i];

                renderingMesh.positions = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.positions.size() * sizeof(glm::vec3),
                    (void *)(mesh.positions.data())
                );
                renderingMesh.normals = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.normals.size() * sizeof(glm::vec3),
                    (void *)(mesh.normals.data())
                );
                renderingMesh.texcoords = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.texcoords.size() * sizeof(glm::vec2),
                    (void *)(mesh.texcoords.data())
                );
                renderingMesh.indices = std::make_unique<VkIBO>(
                    aContext, aAllocator,
                    mesh.indices.size() * sizeof(uint32_t),
                    mesh.indices.size(), 
                    (void *)(mesh.indices.data())
                );
            }
        }

        void upload(VkCommandBuffer uploadCmd) {
            for (auto& mesh : meshes) {
                mesh.positions->upload(uploadCmd);
                mesh.normals->upload(uploadCmd);
                mesh.texcoords->upload(uploadCmd);
                mesh.indices->upload(uploadCmd);
            }
        }

        PipeLineGenerator bindPipeLine(PipeLineGenerator aGenerator) {
            aGenerator
            // positions
            .addVertexInfo(0, 0, sizeof(float) * 3, VK_FORMAT_R32G32B32_SFLOAT)
            // normals
            .addVertexInfo(1, 1, sizeof(float) * 3, VK_FORMAT_R32G32B32_SFLOAT)
            // texcoords
            .addVertexInfo(2, 2, sizeof(float) * 2, VK_FORMAT_R32G32_SFLOAT);;
            return aGenerator;
        }

        void onDraw(VkCommandBuffer cmd) {
            for (auto& mesh : meshes) {
                VkBuffer buffers[3] = { mesh.positions->get(), mesh.normals->get(), mesh.texcoords->get()};
                VkDeviceSize offsets[3] = { 0, 0, 0 };
                vkCmdBindVertexBuffers(cmd, 0, 3, buffers, offsets);
                mesh.indices->bind(cmd);
                mesh.indices->draw(cmd);
            }
        }

        static void uploadScope(lut::VulkanContext const& aContext, const std::function<void(VkCommandBuffer)> &cb) {
            labutils::CommandPool tempPool = labutils::create_command_pool(aContext, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
            // create upload cmd buffer, fence
            labutils::Fence uploadComplete = labutils::create_fence(aContext);
            VkCommandBuffer uploadCmd = labutils::alloc_command_buffer(aContext, tempPool.handle);

            // begin cmd
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = nullptr;

            if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
                VK_SUCCESS != res) {
                throw lut::Error("Beginning command buffer recording\n"
                    "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
            }

            // record upload cmd
            cb(uploadCmd);

            // end cmd
            if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res) {
                throw lut::Error("Ending command buffer recording\n"
                    "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
            }
            // submit upload cmd
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &uploadCmd;

            if (auto const res = vkQueueSubmit(
                aContext.graphicsQueue, 
                1, &submitInfo, 
                uploadComplete.handle);
                VK_SUCCESS != res) {
                throw lut::Error( "Submitting commands\n" 
                    "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
            }

            // Wait for commands to finish before we destroy the temporary resources
            if( auto const res = vkWaitForFences( aContext.device, 1, &uploadComplete.handle,
                VK_TRUE, std::numeric_limits<std::uint64_t>::max() ); VK_SUCCESS != res ) {
                throw lut::Error( "Waiting for upload to complete\n" 
                    "vkWaitForFences() returned %s", lut::to_string(res).c_str());
            }
            vkFreeCommandBuffers( aContext.device, tempPool.handle, 1, &uploadCmd );
        }

    private:
        std::vector<RenderingMesh> meshes;
    };
}