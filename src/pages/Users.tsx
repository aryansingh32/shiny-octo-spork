import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getUsersFromBackend, createUserInBackend, updateUserInBackend, deleteUserFromBackend } from '@/services/backendApi';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { AlertCircle, Plus, Edit, Trash2 } from 'lucide-react';
import { toast } from 'sonner';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';

type UserFormData = {
  name: string;
  age: number;
  address: string;
  contactPhone: string;
  emergencyContact: string;
  medicalConditions: string;
};

const Users = () => {
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<any>(null);
  const [formData, setFormData] = useState<UserFormData>({
    name: '',
    age: 65,
    address: '',
    contactPhone: '',
    emergencyContact: '',
    medicalConditions: '',
  });

  const queryClient = useQueryClient();

  // Fetch users
  const { data: users, isLoading, error } = useQuery({
    queryKey: ['users'],
    queryFn: getUsersFromBackend,
  });

  // Add user mutation
  const addUserMutation = useMutation({
    mutationFn: createUserInBackend,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      toast.success('User added successfully');
      setIsAddDialogOpen(false);
      resetForm();
    },
    onError: (error) => {
      toast.error(`Failed to add user: ${error instanceof Error ? error.message : 'Unknown error'}`);
    },
  });

  // Update user mutation
  const updateUserMutation = useMutation({
    mutationFn: ({ id, userData }: { id: string; userData: any }) =>
      updateUserInBackend(id, userData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      toast.success('User updated successfully');
      setIsEditDialogOpen(false);
      resetForm();
    },
    onError: (error) => {
      toast.error(`Failed to update user: ${error instanceof Error ? error.message : 'Unknown error'}`);
    },
  });

  // Delete user mutation
  const deleteUserMutation = useMutation({
    mutationFn: deleteUserFromBackend,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      toast.success('User deleted successfully');
    },
    onError: (error) => {
      toast.error(`Failed to delete user: ${error instanceof Error ? error.message : 'Unknown error'}`);
    },
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'age' ? parseInt(value) || 0 : value,
    }));
  };

  const handleAddUser = () => {
    const userData = {
      ...formData,
      medicalConditions: formData.medicalConditions.split(',').map(s => s.trim()),
      preferences: {
        reminderNotifications: 'both',
        fontSize: 'large',
      },
      photo: '/placeholder.svg',
    };

    addUserMutation.mutate(userData as any);
  };

  const handleUpdateUser = () => {
    if (!selectedUser) return;
    
    const userData = {
      ...formData,
      medicalConditions: formData.medicalConditions.split(',').map(s => s.trim()),
    };

    updateUserMutation.mutate({ id: selectedUser.id, userData });
  };

  const handleDeleteUser = (userId: string) => {
    if (window.confirm('Are you sure you want to delete this user?')) {
      deleteUserMutation.mutate(userId);
    }
  };

  const openEditDialog = (user: any) => {
    setSelectedUser(user);
    setFormData({
      name: user.name,
      age: user.age,
      address: user.address || '',
      contactPhone: user.contactPhone || '',
      emergencyContact: user.emergencyContact || '',
      medicalConditions: user.medicalConditions ? user.medicalConditions.join(', ') : '',
    });
    setIsEditDialogOpen(true);
  };

  const resetForm = () => {
    setFormData({
      name: '',
      age: 65,
      address: '',
      contactPhone: '',
      emergencyContact: '',
      medicalConditions: '',
    });
    setSelectedUser(null);
  };

  if (error) {
    return (
      <div className="p-6">
        <Card className="border-destructive">
          <CardHeader>
            <div className="flex items-center">
              <AlertCircle className="h-6 w-6 text-destructive mr-2" />
              <CardTitle>Error Loading Users</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p>{error instanceof Error ? error.message : 'Failed to load users'}</p>
          </CardContent>
          <CardFooter>
            <Button
              variant="outline"
              onClick={() => queryClient.invalidateQueries({ queryKey: ['users'] })}
            >
              Retry
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">User Management</h1>
        <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add User
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New User</DialogTitle>
              <DialogDescription>
                Add a new user to the elderly care system.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="name" className="text-right">
                  Name
                </Label>
                <Input
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="age" className="text-right">
                  Age
                </Label>
                <Input
                  id="age"
                  name="age"
                  type="number"
                  value={formData.age}
                  onChange={handleInputChange}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="address" className="text-right">
                  Address
                </Label>
                <Input
                  id="address"
                  name="address"
                  value={formData.address}
                  onChange={handleInputChange}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="contactPhone" className="text-right">
                  Phone
                </Label>
                <Input
                  id="contactPhone"
                  name="contactPhone"
                  value={formData.contactPhone}
                  onChange={handleInputChange}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="emergencyContact" className="text-right">
                  Emergency Contact
                </Label>
                <Input
                  id="emergencyContact"
                  name="emergencyContact"
                  value={formData.emergencyContact}
                  onChange={handleInputChange}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="medicalConditions" className="text-right">
                  Medical Conditions
                </Label>
                <Input
                  id="medicalConditions"
                  name="medicalConditions"
                  value={formData.medicalConditions}
                  onChange={handleInputChange}
                  className="col-span-3"
                  placeholder="Separate with commas"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleAddUser}>Add User</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Users</CardTitle>
          <CardDescription>
            Manage users in the elderly care monitoring system
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex justify-center items-center h-32">
              <p>Loading users...</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Age</TableHead>
                  <TableHead>Contact</TableHead>
                  <TableHead>Medical Conditions</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users && users.length > 0 ? (
                  users.map((user) => (
                    <TableRow key={user.id}>
                      <TableCell className="font-medium">{user.name}</TableCell>
                      <TableCell>{user.age}</TableCell>
                      <TableCell>
                        {user.contactPhone && (
                          <div className="text-sm">
                            <span className="font-semibold">Phone:</span> {user.contactPhone}
                          </div>
                        )}
                        {user.emergencyContact && (
                          <div className="text-sm">
                            <span className="font-semibold">Emergency:</span> {user.emergencyContact}
                          </div>
                        )}
                        {!user.contactPhone && !user.emergencyContact && (
                          <span className="text-muted-foreground">No contact info</span>
                        )}
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {user.medicalConditions && user.medicalConditions.length > 0 ? (
                            user.medicalConditions.map((condition, idx) => (
                              <Badge key={idx} variant="outline">
                                {condition}
                              </Badge>
                            ))
                          ) : (
                            <span className="text-muted-foreground">No conditions</span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="outline"
                          size="icon"
                          className="mr-2"
                          onClick={() => openEditDialog(user)}
                        >
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="destructive"
                          size="icon"
                          onClick={() => handleDeleteUser(user.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center">
                      No users found. Add a user to get started.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit User</DialogTitle>
            <DialogDescription>
              Update user information.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="edit-name" className="text-right">
                Name
              </Label>
              <Input
                id="edit-name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="edit-age" className="text-right">
                Age
              </Label>
              <Input
                id="edit-age"
                name="age"
                type="number"
                value={formData.age}
                onChange={handleInputChange}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="edit-address" className="text-right">
                Address
              </Label>
              <Input
                id="edit-address"
                name="address"
                value={formData.address}
                onChange={handleInputChange}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="edit-contactPhone" className="text-right">
                Phone
              </Label>
              <Input
                id="edit-contactPhone"
                name="contactPhone"
                value={formData.contactPhone}
                onChange={handleInputChange}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="edit-emergencyContact" className="text-right">
                Emergency Contact
              </Label>
              <Input
                id="edit-emergencyContact"
                name="emergencyContact"
                value={formData.emergencyContact}
                onChange={handleInputChange}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="edit-medicalConditions" className="text-right">
                Medical Conditions
              </Label>
              <Input
                id="edit-medicalConditions"
                name="medicalConditions"
                value={formData.medicalConditions}
                onChange={handleInputChange}
                className="col-span-3"
                placeholder="Separate with commas"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleUpdateUser}>Update User</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Users;
