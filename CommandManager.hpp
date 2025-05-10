#pragma once
#include "Task.hpp"

class OperationController;

class CommandManager {
public:
    explicit CommandManager(OperationController& ctrl) : ctl(ctrl) {}
    void execute(Task& task) const;
private:
    OperationController& ctl;
};
