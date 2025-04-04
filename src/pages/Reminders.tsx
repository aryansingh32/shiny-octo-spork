import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getRemindersFromBackend,
  getUsersFromBackend,
  addReminder,
  updateReminderInBackend,
  acknowledgeReminder,
  getUserRemindersFromBackend,
} from "@/services/backendApi";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import { AlertCircle, Check, Clock, Plus } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";
import VoiceReminder from "@/components/VoiceReminders";

const reminderTypes = ["medication", "appointment", "activity", "other"];
const reminderPriorities = ["low", "medium", "high"];
const reminderRecurrences = ["once", "daily", "weekly", "monthly"];

const Reminders = () => {
  const queryClient = useQueryClient();
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [selectedUserId, setSelectedUserId] = useState<string>("");
  const [formData, setFormData] = useState({
    userId: "",
    title: "",
    description: "",
    dateTime: "",
    type: "medication",
    priority: "medium",
    recurrence: "once",
  });

  // Fetch users for the dropdown
  const { data: users } = useQuery({
    queryKey: ["users"],
    queryFn: getUsersFromBackend,
  });

  // Fetch reminders for the selected user
  const { data: reminders, isLoading, error, refetch } = useQuery({
    queryKey: ["reminders", selectedUserId],
    queryFn: async () => {
      console.log("Fetching reminders with selectedUserId:", selectedUserId);
      if (!selectedUserId || selectedUserId === "all") {
        const allReminders = await getRemindersFromBackend();
        console.log("All reminders:", allReminders);
        return allReminders;
      }
      const userReminders = await getUserRemindersFromBackend(selectedUserId);
      console.log("User reminders:", userReminders);
      return userReminders;
    },
    enabled: true, // Always enabled
  });

  // Add reminder mutation
  const addReminderMutation = useMutation({
    mutationFn: ({ userId, reminderData }: { userId: string; reminderData: any }) =>
      addReminder(userId, reminderData),
    onSuccess: () => {
      console.log("Reminder added successfully, invalidating queries...");
      // Invalidate and refetch both queries
      queryClient.invalidateQueries({ queryKey: ["reminders"] });
      queryClient.invalidateQueries({ queryKey: ["reminders", selectedUserId] });
      refetch(); // Force a refetch
      setIsAddDialogOpen(false);
      resetForm();
      toast.success("Reminder added successfully");
    },
    onError: (error) => {
      console.error("Error adding reminder:", error);
      toast.error(`Failed to add reminder: ${error instanceof Error ? error.message : "Unknown error"}`);
    },
  });

  // Complete reminder mutation
  const completeReminderMutation = useMutation({
    mutationFn: ({ userId, reminderId }: { userId: string; reminderId: string }) =>
      updateReminderInBackend(userId, reminderId, { 
        completed: true,
        is_acknowledged: 1,
        status: "completed",
        acknowledged_time: new Date().toISOString().slice(0, 19).replace('T', ' ')
      }),
    onSuccess: () => {
      // Invalidate all possible queries
      queryClient.invalidateQueries({ queryKey: ["reminders"] });
      queryClient.invalidateQueries({ queryKey: ["reminders", selectedUserId] });
      refetch(); // Force a refetch
      toast.success("Reminder marked as completed");
    },
    onError: (error) => {
      toast.error(`Failed to complete reminder: ${error instanceof Error ? error.message : "Unknown error"}`);
    },
  });

  // Acknowledge reminder mutation
  const acknowledgeReminderMutation = useMutation({
    mutationFn: ({ userId, reminderId }: { userId: string; reminderId: string }) =>
      acknowledgeReminder(userId, reminderId),
    onSuccess: () => {
      // Invalidate all possible queries
      queryClient.invalidateQueries({ queryKey: ["reminders"] });
      queryClient.invalidateQueries({ queryKey: ["reminders", selectedUserId] });
      refetch(); // Force a refetch
      toast.success("Reminder acknowledged");
    },
    onError: (error) => {
      toast.error(`Failed to acknowledge reminder: ${error instanceof Error ? error.message : "Unknown error"}`);
    },
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSelectChange = (name: string, value: string) => {
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleAddReminder = () => {
    const { userId, dateTime, type, title, description, priority, recurrence } = formData;
    
    if (!userId) {
      toast.error("Please select a user");
      return;
    }

    if (!dateTime) {
      toast.error("Please select a date and time");
      return;
    }

    if (!title) {
      toast.error("Please enter a title");
      return;
    }

    // Convert the datetime-local input to the format expected by the backend
    const formattedTime = new Date(dateTime).toISOString().slice(0, 16).replace('T', ' ');

    addReminderMutation.mutate({
      userId,
      reminderData: {
        title,
        time: formattedTime,
        type,
        description,
        priority: priority || 1,
        recurrence,
      },
    });
  };

  const handleCompleteReminder = (reminder: any) => {
    completeReminderMutation.mutate({
      userId: reminder.user_id || reminder.userId,
      reminderId: reminder.id,
    });
  };

  const handleAcknowledgeReminder = (reminder: any) => {
    acknowledgeReminderMutation.mutate({
      userId: reminder.user_id || reminder.userId,
      reminderId: reminder.id,
    });
  };

  const resetForm = () => {
    setFormData({
      userId: "",
      title: "",
      description: "",
      dateTime: "",
      type: "medication",
      priority: "medium",
      recurrence: "once",
    });
  };

  const getUsername = (userId: string) => {
    if (!userId) return "Unknown User";
    
    const user = users?.find((user) => user.id === userId);
    return user ? user.name : "Unknown User";
  };

  const getPriorityBadgeVariant = (priority: string) => {
    switch (priority) {
      case "high":
        return "destructive";
      case "medium":
        return "default";
      default:
        return "secondary";
    }
  };

  const formatDateTime = (dateTimeStr: string) => {
    if (!dateTimeStr) return "Not specified";
    
    try {
      return format(new Date(dateTimeStr), "MMM d, yyyy h:mm a");
    } catch (e) {
      console.error("Error formatting date:", e, dateTimeStr);
      return dateTimeStr;
    }
  };

  if (error) {
    return (
      <div className="p-6">
        <Card className="border-destructive">
          <CardHeader>
            <div className="flex items-center">
              <AlertCircle className="h-6 w-6 text-destructive mr-2" />
              <CardTitle>Error Loading Reminders</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p>{error instanceof Error ? error.message : "Failed to load reminders"}</p>
          </CardContent>
          <CardContent>
            <Button
              variant="outline"
              onClick={() => queryClient.invalidateQueries({ queryKey: ["reminders"] })}
            >
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Reminders</h1>
        <div className="flex gap-2">
          <Select
            value={selectedUserId}
            onValueChange={setSelectedUserId}
          >
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Filter by user" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Users</SelectItem>
              {users?.map((user) => (
                <SelectItem key={user.id} value={user.id.toString()}>
                  {user.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <VoiceReminder 
            userId={selectedUserId !== "all" ? parseInt(selectedUserId) : undefined}
            defaultMessage="Hello! This is a custom voice reminder from the elderly care system."
          />
          <Button onClick={() => setIsAddDialogOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Add Reminder
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>All Reminders</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex justify-center items-center h-32">
              <p>Loading reminders...</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>User</TableHead>
                  <TableHead>Title</TableHead>
                  <TableHead>When</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Priority</TableHead>
                  <TableHead>Recurrence</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {reminders && reminders.length > 0 ? (
                  reminders.map((reminder) => (
                    <TableRow key={reminder.id}>
                      <TableCell>{reminder.user_name || getUsername(reminder.user_id) || "Unknown User"}</TableCell>
                      <TableCell className="font-medium">{reminder.title}</TableCell>
                      <TableCell>{formatDateTime(reminder.scheduled_time || reminder.dateTime)}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{reminder.reminder_type || reminder.type || "Unknown"}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={getPriorityBadgeVariant(reminder.priority)}>
                          {reminder.priority || "medium"}
                        </Badge>
                      </TableCell>
                      <TableCell>{reminder.recurrence || "once"}</TableCell>
                      <TableCell>
                        {reminder.status === "completed" || reminder.is_acknowledged === 1 ? (
                          <Badge variant="outline" className="bg-green-100 text-green-800">
                            Completed
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="bg-amber-100 text-amber-800">
                            {reminder.status || "Pending"}
                          </Badge>
                        )}
                        {process.env.NODE_ENV === 'development' && (
                          <div className="text-xs text-gray-500 mt-1">
                            status: {reminder.status}, is_acknowledged: {reminder.is_acknowledged}
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        {(reminder.status !== "completed" && reminder.is_acknowledged !== 1) && (
                          <div className="flex space-x-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleCompleteReminder(reminder)}
                            >
                              <Check className="h-4 w-4 mr-1" />
                              Complete
                            </Button>
                            <VoiceReminder 
                              reminderId={reminder.id} 
                              userId={reminder.user_id || reminder.userId}
                              defaultMessage={reminder.title}
                              compact={true} 
                            />
                          </div>
                        )}
                        {(reminder.status === "completed" || reminder.is_acknowledged === 1) && (
                          <VoiceReminder 
                            reminderId={reminder.id} 
                            userId={reminder.user_id || reminder.userId}
                            defaultMessage={reminder.title}
                            compact={true} 
                          />
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center">
                      No reminders found. Add a reminder to get started.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

        <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Reminder
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New Reminder</DialogTitle>
              <DialogDescription>
                Create a new reminder for an elderly individual.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="userId" className="text-right">
                  User
                </Label>
                <div className="col-span-3">
                  <Select
                    value={formData.userId}
                    onValueChange={(value) => handleSelectChange("userId", value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select user" />
                    </SelectTrigger>
                    <SelectContent>
                      {users?.map((user) => (
                        <SelectItem key={user.id} value={user.id}>
                          {user.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="title" className="text-right">
                  Title
                </Label>
                <Input
                  id="title"
                  name="title"
                  value={formData.title}
                  onChange={handleInputChange}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="description" className="text-right">
                  Description
                </Label>
                <Input
                  id="description"
                  name="description"
                  value={formData.description}
                  onChange={handleInputChange}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="dateTime" className="text-right">
                  Date & Time
                </Label>
                <Input
                  id="dateTime"
                  name="dateTime"
                  type="datetime-local"
                  value={formData.dateTime}
                  onChange={handleInputChange}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="type" className="text-right">
                  Type
                </Label>
                <div className="col-span-3">
                  <Select
                    value={formData.type}
                    onValueChange={(value) => handleSelectChange("type", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {reminderTypes.map((type) => (
                        <SelectItem key={type} value={type}>
                          {type.charAt(0).toUpperCase() + type.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="priority" className="text-right">
                  Priority
                </Label>
                <div className="col-span-3">
                  <Select
                    value={formData.priority}
                    onValueChange={(value) => handleSelectChange("priority", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {reminderPriorities.map((priority) => (
                        <SelectItem key={priority} value={priority}>
                          {priority.charAt(0).toUpperCase() + priority.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="recurrence" className="text-right">
                  Recurrence
                </Label>
                <div className="col-span-3">
                  <Select
                    value={formData.recurrence}
                    onValueChange={(value) => handleSelectChange("recurrence", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {reminderRecurrences.map((recurrence) => (
                        <SelectItem key={recurrence} value={recurrence}>
                          {recurrence.charAt(0).toUpperCase() + recurrence.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleAddReminder}>Add Reminder</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
    </div>
  );
};

export default Reminders;
