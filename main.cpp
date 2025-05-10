#include "CLIParser.hpp"
#include "CommandManager.hpp"
#include "OperationController.hpp"

int main(int argc, char* argv[])
{
    CLIParser            cli;
    auto maybeTask = cli.parse(argc, argv);
    if (!maybeTask) return 1;

    OperationController  oc;
    CommandManager       mgr(oc);

    mgr.execute(*maybeTask);
    return 0;
}
