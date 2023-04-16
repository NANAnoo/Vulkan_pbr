#include "vkutil.hpp"

#include <vector>

#include <cstdio>
#include <cassert>

#include "error.hpp"
#include "to_string.hpp"

namespace labutils
{
	ShaderModule load_shader_module( VulkanContext const& aContext, char const* aSpirvPath )
	{
		throw Error( "Not yet implemented" ); //TODO- (Section 1/Exercise 2) implement me!
	}


	CommandPool create_command_pool( VulkanContext const& aContext, VkCommandPoolCreateFlags aFlags )
	{
		throw Error( "Not yet implemented" ); //TODO: implement me!

	}

	VkCommandBuffer alloc_command_buffer( VulkanContext const& aContext, VkCommandPool aCmdPool )
	{
		throw Error( "Not yet implemented" ); //TODO: implement me!
	}


	Fence create_fence( VulkanContext const& aContext, VkFenceCreateFlags aFlags )
	{
		throw Error( "Not yet implemented" ); //TODO: implement me!
	}

	Semaphore create_semaphore( VulkanContext const& aContext )
	{
		throw Error( "Not yet implemented" ); //TODO: implement me!
	}




}
