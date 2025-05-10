#include "CommandManager.hpp"
#include "OperationController.hpp"

void CommandManager::execute(Task& t) const
{
    ctl.runTask(t);
}
